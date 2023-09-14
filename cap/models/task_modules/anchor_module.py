# Copyright (c) Changan Auto. All rights reserved.

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from cap.registry import OBJECT_REGISTRY
from .output_module import OutputModule


@OBJECT_REGISTRY.register
class AnchorModule(OutputModule):
    """The container of Anchor-based detector module.

    This class serves as the container of anchor-based detector,
    which takes feature maps as input and outputs predictions which
    applies onto self-generated anchor boxes.

    All the actaul calculations are implemented in component modules.

    Args:
        anchor_generator: Anchor generator module, generates anchors
            to which the offsets are applied.
        head: Anchor head network, transfroms input feature maps
            into predictions (like regression map and classification
            score in RPN).
        ext_feat: Extra feature module that processes input feature maps
            before head.
        target: Target generator module, generates training target given
            ground truth labels and anchor boxes.
        loss: Loss module, calculates training loss by comparing head
            predictions with training targets.
        postprocess: Postprocess module, applies predictions generated
            by head module onto anchors to get final prediction.
        desc: Desc module, adds user-defined description to prediction.
        output_head_out: Whether to output raw prediction of head module.
            Mostly used for the purpose of visualization.
        output_target: Wheter to output training target. This argument works
            only when loss is presented. Mostly used to
            calculate metrics.
        target_keys: Keys used to get ground truths from input data. When
            target generator is presented, these data are inputs to target
            generator. Otherwise, the data are directly fed into loss module
            as ground truths.
        target_opt_keys: Keys used to get optional ground truths from input
            data. Only works when target generator is presented.
    """

    def __init__(
        self,
        anchor_generator: nn.Module,
        head: nn.Module,
        ext_feat: Optional[nn.Module] = None,
        target: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        desc: Optional[nn.Module] = None,
        output_head_out: bool = False,
        output_target: bool = False,
        target_keys: Tuple[str] = ("gt_boxes", "gt_boxes_num"),
        target_opt_keys: Tuple[str] = (
            "ig_regions",
            "ig_regions_num",
        ),
    ):
        super().__init__(
            head,
            loss=loss,
            target=target,
            postprocess=postprocess,
            keep_name=True,
        )

        # submodule
        self.ext_feat = ext_feat
        self.anchor_generator = anchor_generator
        self.desc = desc

        self._output_head_out = output_head_out
        self._output_target = output_target
        self._target_keys = target_keys
        self._target_opt_keys = target_opt_keys

    @property
    def with_ext_feat(self) -> bool:
        return self.ext_feat is not None

    def forward(
        self,
        feat_maps: List[torch.Tensor],
        y: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        # optionally process input features with extra layers
        if self.with_ext_feat:
            feat_maps = self.ext_feat(feat_maps)

        # generate anchors
        mlvl_anchors = self.anchor_generator(feat_maps)

        # get prediction scores
        head_out = self.head(feat_maps)

        out_dict = OrderedDict()

        if self._output_head_out:
            out_dict.update(head_out)

        # apply prediction scores to anchors to get final predictions
        if self.with_postprocess:
            pred = self.postprocess(
                mlvl_anchors, head_out, y.get("im_hw", None)
            )
            if self.desc is not None:
                pred = self.desc(pred)
            out_dict.update(pred)

        # calculate loss between predictions and ground truths
        if self.with_loss:
            # generate targets on-the-fly
            if self.has_target:
                _, targets = self.target(
                    mlvl_anchors,
                    *[y[k] for k in self._target_keys],
                    **{k: y.get(k, None) for k in self._target_opt_keys},
                )
            # load targets direclty
            else:
                assert len(self._target_keys) == 1
                targets = y[self._target_keys[0]]

            if self._output_target:
                out_dict.update(targets)

            loss = self.loss(head_out, targets)
            out_dict.update(loss)

        return out_dict
