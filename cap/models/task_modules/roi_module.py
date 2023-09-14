# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY
from .output_module import OutputModule


@OBJECT_REGISTRY.register
class RoIModule(OutputModule):
    """The container of ROIHead module.

    This class serves as the container of ROI Head module, which takes
    feature maps and corresponding ROIs (Region of Interests) as input,
    and outputs ROI-wise predictions.

    All the actual calculations are implemented in component modules.

    Args:
        roi_feat_extractor: ROI feature extractor.
        head: RoI head network, transfroms input feature maps
            into predictions (like regression map and classification
            score in RCNN).
        ext_feat: Extra feature module that processes input feature maps
            before head.
        target: Target generator module, generates training target given
            ground truth labels and rois.
        loss: Loss module, calculates training loss by comparing head
            predictions with training targets.
        postprocess: Postprocess module, applies predictions generated
            by head module onto rois to get final prediction.
        head_desc: Desc module, adds user-defined description to outputs of
            head module.
        post_desc: Desc module, adds user-defined description to outputs of
            postprocess module.
        output_head_out: Whether to output raw prediction of head module.
            Mostly used for the purpose of visualization.
        output_target: Wheter to output training target. This argument works
            only when loss is presented. Mostly used to
            calculate metrics.
        target_keys: Keys used to get ground truths from input data. When
            target generator is presented, these data are inputs to target
            generator. Otherwise, the data are rois and labels directly fed
            into loss module as ground truths, respectively.
        target_opt_keys: Keys used to get optional ground truths from input
            data. Only works when target generator is presented.

    """

    def __init__(
        self,
        roi_feat_extractor: nn.Module,
        head: nn.Module,
        ext_feat: Optional[nn.Module] = None,
        target: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        head_desc: Optional[nn.Module] = None,
        post_desc: Optional[nn.Module] = None,
        output_head_out: bool = False,
        output_target: bool = False,
        roi_key: str = "pred_boxes",
        target_keys: Tuple[str] = ("gt_boxes", "gt_boxes_num"),
        target_opt_keys: Tuple[str] = (
            "ig_regions",
            "ig_regions_num",
        ),
        postprocess_keys: Optional[Tuple[str]] = (),
    ):
        super().__init__(
            head=head,
            loss=loss,
            target=target,
            postprocess=postprocess,
            keep_name=True,
        )

        # submodules
        self.ext_feat = ext_feat
        self.roi_feat_extractor = roi_feat_extractor
        self.head_desc = head_desc
        self.post_desc = post_desc

        self._output_head_out = output_head_out
        self._output_target = output_target
        self._roi_key = roi_key
        self._target_keys = target_keys
        self._target_opt_keys = target_opt_keys
        self._postprocess_keys = postprocess_keys

    @property
    def with_ext_feat(self) -> bool:
        return self.ext_feat is not None

    def forward(
        self,
        feat_maps: List[torch.Tensor],
        rpn_pred: Optional[Dict[str, torch.Tensor]] = None,
        y: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:

        # optionally process input features with extra layers
        if self.with_ext_feat:
            feat_maps = self.ext_feat(feat_maps)

        # rpn_pred presented, check has_target
        if rpn_pred is not None:
            batch_rois = rpn_pred[self._roi_key]
            # two stage training
            if self.has_target:
                batch_rois, labels = self.target(
                    batch_rois,
                    *[y[k] for k in self._target_keys],
                    **{k: y.get(k, None) for k in self._target_opt_keys}
                )
            # inference
            else:
                labels = None
        # roi-only training, get rois and labels directly from input data
        else:
            assert not self.has_target
            assert len(self._target_keys) == 2
            batch_rois = y[self._target_keys[0]]
            labels = y[self._target_keys[1]]

        out_dict = OrderedDict()
        if self._output_target and labels is not None:
            out_dict.update(labels)

        # get roi feature
        roi_feat = self.roi_feat_extractor(feat_maps, batch_rois)

        # get head prediction
        head_out = self.head(roi_feat)
        if self.head_desc is not None:
            head_out = self.head_desc(head_out)

        # import numpy as np
        # np.save(
        #     "vehicle_head_out_cls_pred_0213_2.npy", [it.cpu().detach().numpy() for it in head_out._idata['rcnn_cls_pred']], allow_pickle=True
        # )

        # np.save(
        #     "vehicle_head_out_reg_pred_0213_2.npy", [it.cpu().detach().numpy() for it in head_out._idata['rcnn_reg_pred']], allow_pickle=True
        # )


        if self._output_head_out:
            out_dict.update(head_out)

        if self.with_loss:
            out_dict.update(self.loss(head_out, labels))

        if self.with_postprocess:
            pred = self.postprocess(
                batch_rois,
                head_out,
                **{k: y.get(k, None) for k in self._postprocess_keys}
            )
            if self.post_desc is not None:
                pred = self.post_desc(pred)
            out_dict.update(pred)
        return out_dict
