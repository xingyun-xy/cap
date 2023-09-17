# -*- coding:utf-8 -*-
# Copyright (c) Changan Auto, All rights reserved.

import logging
from typing import Mapping, Optional

import torch
from torch import nn

try:
    import wenet
except ImportError:
    wenet = None
if wenet is None:
    LabelSmoothingLoss = nn.Module
    CTC = nn.Module
    IGNORE_ID = -1
    add_sos_eos = None
    reverse_pad_list = None
    th_accuracy = None
else:
    from wenet.transformer.ctc import CTC
    from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
    from wenet.utils.common import (
        IGNORE_ID,
        add_sos_eos,
        reverse_pad_list,
        th_accuracy,
    )

from cap.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class CocktailE2EStructureA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        vfea_extractor: nn.Module,
        afea_extractor: Optional[nn.Module],
        encoder: nn.Module,
        decoder: nn.Module,
        ctc: Optional[CTC] = None,
        ctc_weight: Optional[float] = 0.5,
        ignore_id: int = IGNORE_ID,
        att_loss: Optional[LabelSmoothingLoss] = None,
        reverse_weight: Optional[float] = 0.0,
    ):
        super().__init__()
        if wenet is None:
            raise ModuleNotFoundError(
                "wenet is required by ``CocktailE2EStructureA``"
            )
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        # 设置 sos 和 eos, eos 和 sos 相同, 都是词典的最后一位
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        # 处理 ctc 和 ctc loss weight
        self.ctc = ctc
        if self.ctc is not None:
            self.ctc_weight = ctc_weight
            assert 0.0 <= self.ctc_weight <= 1.0, ctc_weight
        self.att_loss = att_loss
        self.reverse_weight = reverse_weight
        # 指定各个模块
        self.vfea_extractor = vfea_extractor
        self.afea_extractor = afea_extractor
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data: Mapping[str, Optional[torch.Tensor]]):
        # 预处理图片序列
        images = data["images"]
        logging.debug(f"images min: {images.min()}, max: {images.max()}")
        logging.debug(f"input images shape: {images.shape}")
        b, t, c, h, w = images.shape
        images = data["images"].view(-1, c, h, w)
        v_lens = data["images_lens"]
        # 生成视觉特征
        vfea = self.vfea_extractor(images)
        vfea = vfea.view(b, t, -1)
        # 生成音频特征
        afea = data["audio"]
        a_lens = data["audio_lens"]
        if self.afea_extractor is not None:
            afea = self.afea_extractor(afea)
        # 过 Encoder 模型
        a_b, a_t, _ = afea.shape
        encoder_out, encoder_mask = self.encoder(afea, vfea, v_lens, a_lens)
        #  获取标签和标签长度
        ys, ys_lens = data["label"], data["label_length"]
        # 修改label的标签
        ys_in, ys_out = add_sos_eos(ys, self.sos, self.eos, self.ignore_id)
        # 如果reverse 解码的权重 >0.0 的话
        if self.reverse_weight > 0.0:
            r_ys = reverse_pad_list(ys, ys_lens, float(self.ignore_id))
            r_ys_in, r_ys_out = add_sos_eos(
                r_ys, self.sos, self.eos, self.ignore_id
            )
            inputs = [
                ys_in,
                ys_lens + 1,
                r_ys_in,
                self.reverse_weight,
            ]
        else:
            inputs = [ys_in, ys_lens + 1, None, 0]
        # 前向 decoder_out
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, *inputs
        )
        # 计算 attention loss
        loss_att = self.att_loss(decoder_out, ys_out)
        # 计算 reverse attention loss
        if self.reverse_weight > 0.0:
            r_loss_att = self.att_loss(r_decoder_out, r_ys_out)
            loss_att = (
                loss_att * (1 - self.reverse_weight)
                + r_loss_att * self.reverse_weight
            )
        # 计算CTC Loss
        if self.ctc is not None:
            encoder_out_lens = encoder_mask.squeeze(1).sum(1)
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys, ys_lens)
            loss = (
                loss_att * (1 - self.ctc_weight) + loss_ctc * self.ctc_weight
            )
        else:
            loss = loss_att
            loss_ctc = None
        # 计算 attention 结果的 Accuracy
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out,
            ignore_label=self.ignore_id,
        )
        return loss, loss_att, loss_ctc, acc_att
