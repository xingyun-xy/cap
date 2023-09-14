# Copyright (c) Changan Auto. All rights reserved.

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cap.metrics.metric_3dv_utils import bev3d_nms
from cap.models.base_modules.postprocess import ArgmaxPostprocess
from cap.models.losses.real3d_losses import sigmoid_and_clip
from cap.registry import OBJECT_REGISTRY

__all__ = ["BEVPostprocess", "BEV3Decoder"]


@OBJECT_REGISTRY.register
class BEVPostprocess(object):
    """Apply argmax of segs and bev_segs in predict dict.

    Args:
        seg_name (str): seg name of data to apply argmax.
        bev_seg_name (str): bev_seg name of data to apply argmax.

        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not.

    """

    def __init__(
        self, seg_name: str, bev_seg_name: str, dim: int, keepdim: bool = False
    ):
        self.seg_name = seg_name
        self.bev_seg_name = bev_seg_name
        self.seg_argmax = ArgmaxPostprocess(seg_name, dim, keepdim)
        self.bev_seg_argmax = ArgmaxPostprocess(bev_seg_name, dim, keepdim)

    def __call__(self, pred: Mapping, *args, **kwargs):
        if self.seg_name in pred:
            pred = self.seg_argmax(pred)
        if self.bev_seg_name in pred:
            pred = self.bev_seg_argmax(pred)
        return pred


@OBJECT_REGISTRY.register
class BEV3Decoder(object):
    """The Bev3D Decoder, convert the bev3d output to obj results.

    Args:
        cls_dimension (np.ndarray): the average dimension of each category.
        nms_thresh (float): the iou threshold for nms.
        topk (int): The maximum number of objects in multi-view bev
            defualt=100.
        max_pool_kernel (int): The max pooling kernel that is used
            to do the nms, default=9.
        vcs_range (sequence of float): vcs range.(order is
            (bottom,right,top,left)).
        bev_3d_out_size (sequence of float): output size of bev3d head.
        num_classes (int): num of classes.
        bev_nms (bool): whether to use the nms, default: False
        add_hm_eps (bool): whether to add eps for heatmap, default: False.
            NOTE: since the quantize of heatmap will cause the same score
            of points in the maxpooling kernel region, will resulting high
            score FP objs in eval. Add eps to make them difference.

    """

    def __init__(
        self,
        cls_dimension: np.ndarray,
        nms_thresh: float,
        topk: Optional[int] = 100,
        max_pool_kernel: Optional[int] = 3,
        vcs_range: Optional[Sequence] = (-30.0, -51.2, 72.4, 51.2),
        bev_3d_out_size: Optional[Sequence] = (256, 256),
        num_classes: int = 1,
        bev_nms: bool = False,
        add_hm_eps: bool = False,
    ):
        self.topk = topk
        self.max_pool_kernel = max_pool_kernel
        self.cls_dimension = cls_dimension
        self.vcs_range = vcs_range
        self.bev_nms = bev_nms
        self.nms_thresh = nms_thresh
        self.add_hm_eps = add_hm_eps
        if self.add_hm_eps:
            torch.manual_seed(1)
            self.hm_eps = (
                torch.rand((1, num_classes, *bev_3d_out_size)) * 1e-3
            )  # (bs, cls, h, w)

    def coord_transform(
        self,
        bev_center: torch.Tensor,
        bev_size: Tuple = (256, 256),
        vcs_range: Tuple = (-30.0, -51.2, 72.4, 51.2),
    ):
        """Convert the bev image coordinate to vcs coordinate.

        Args:
            bev_center (torch.Tensor): the predict object center on bev image
                coordinate. shape: [bs, topk, 2], the "2" means image coord
                [x(u),y(v)].
            bev_size (Tuple, optional): the bev map size. Defaults to (256,256)
            vcs_range (Tuple, optional): vcs visiable range, (order:
                (bottom,right,up,left)), Defaults to (-30.0, -51.2, 72.4, 51.2)
        Returns:
            (torch.Tensor): vcs center, shape:[bs, topk, 2], the "2" means
                vcs coord [x,y].
        """
        m_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_size[0],
            abs(vcs_range[3] - vcs_range[1]) / bev_size[1],
        )  # bev coord [y, x]

        vcs_x = vcs_range[2] - (bev_center[:, :, 1]) * m_perpixel[0]
        vcs_y = vcs_range[3] - (bev_center[:, :, 0]) * m_perpixel[1]
        vcs_center = torch.stack([vcs_x, vcs_y], dim=-1)

        return vcs_center

    def __call__(
        self,
        pred: Mapping,
        label: Dict = None,
    ) -> Dict:

        bev3d_hm = sigmoid_and_clip(pred["bev3d_hm"])
        if self.add_hm_eps:
            bev3d_hm += self.hm_eps.to(bev3d_hm.device)
        # max_pooling nms
        batch_size, num_classes, height, width = bev3d_hm.shape
        kernel = self.max_pool_kernel
        pad = (kernel - 1) // 2
        max_bev3d_hm = F.max_pool2d(
            bev3d_hm, (kernel, kernel), stride=1, padding=(pad, pad)
        )
        bev3d_hm *= (max_bev3d_hm == bev3d_hm).float()
        pred["bev3d_hm"] = bev3d_hm

        for key, out in pred.items():
            channel_size = out.shape[1]
            pred[key] = out.permute(0, 2, 3, 1).contiguous()
            pred[key] = pred[key].view(batch_size, -1, channel_size)

        # get the topk indices of heatmap
        bev3d_hm = pred.pop("bev3d_hm")
        scores, classes = bev3d_hm.max(dim=-1)
        scores, topk_indices = scores.topk(k=self.topk, dim=1)
        pred["bev3d_cls_id"] = classes

        # get the topk indices of regression items (dim, rot etc.)
        for key, out in pred.items():
            pred[key] = torch.cat(
                [_out[_inds][None] for _out, _inds in zip(out, topk_indices)],
                dim=0,
            )

        # convert topk indices value to the coordinate
        pred["bev3d_score"] = scores
        u, v = topk_indices % width, topk_indices // width  # bev: u=x, v=y
        bev3d_center = torch.cat(
            [u[:, :, None], v[:, :, None]], axis=-1
        )  # (u,v) = (x,y)

        if "bev3d_ct_offset" in pred:
            bev3d_ct_offset = pred.pop("bev3d_ct_offset")
        else:
            bev3d_ct_offset = (
                torch.zeros_like(bev3d_center, dtype=torch.float32) + 0.1
            )
        bev3d_center = bev3d_center + bev3d_ct_offset  # (u,v)

        # convert the bev center in bev coor to vcs coor
        vcs_center = self.coord_transform(
            bev3d_center,
            bev_size=(height, width),
            vcs_range=self.vcs_range,
        )
        pred["bev3d_ct"] = vcs_center

        # convert the rot to rad value (sin / cos), range=[-pi,pi]
        bev3d_rot = pred.pop("bev3d_rot")
        rot = torch.atan2(bev3d_rot[:, :, 1], bev3d_rot[:, :, 0])
        pred["bev3d_rot"] = rot

        # squeeze the shape of loc_z from [b,topk,1] -> [b,topk]
        pred["bev3d_loc_z"] = pred["bev3d_loc_z"].squeeze(-1)

        # if cls_dimension is not None, process the pred residual
        # dim to real dim.
        if self.cls_dimension is not None:
            cls_dimension = torch.from_numpy(self.cls_dimension).to(
                pred["bev3d_dim"].device
            )
            average_dim = cls_dimension[pred["bev3d_cls_id"]]
            pred["bev3d_dim"] = torch.exp(pred["bev3d_dim"]) * average_dim

        if self.bev_nms:
            # using vcs rotate_3d iou for matching
            bev3d_bboxes = torch.cat(
                (
                    pred["bev3d_ct"],
                    pred["bev3d_loc_z"].unsqueeze(-1),
                    pred["bev3d_dim"],
                    pred["bev3d_rot"].unsqueeze(-1),
                ),
                dim=2,
            )
            bev3d_score = pred["bev3d_score"]
            bev3d_cls_id = pred["bev3d_cls_id"]

            bev3d_bboxes_nms = []
            bev3d_score_nms = []
            bev3d_cls_id_nms = []

            batch_size = bev3d_bboxes.shape[0]
            for bs in range(batch_size):
                bboxes = []
                scores = []
                labels = []

                _bev3d_bbox = bev3d_bboxes[bs]
                _bev3d_score = bev3d_score[bs]
                _bev3d_cls_id = bev3d_cls_id[bs]
                for i in range(0, num_classes):
                    # get bboxes and scores of this class
                    cls_inds = torch.bitwise_and(
                        (_bev3d_cls_id == i), (_bev3d_score > 0.0)
                    )

                    if not cls_inds.any():
                        continue

                    _scores = _bev3d_score[cls_inds]
                    _cls_id = _bev3d_cls_id[cls_inds]
                    _bboxes_for_nms = _bev3d_bbox[cls_inds, :]
                    selected = bev3d_nms(
                        _bboxes_for_nms, _scores, thresh=self.nms_thresh
                    )
                    remain_bboxes = _bboxes_for_nms[selected, :]
                    remain_scores = _scores[selected]
                    remain_cls_id = _cls_id[selected]

                    bboxes.append(remain_bboxes)
                    scores.append(remain_scores)
                    labels.append(remain_cls_id)

                if bboxes:
                    bboxes = torch.cat(bboxes, dim=0)
                    scores = torch.cat(scores, dim=0)
                    labels = torch.cat(labels, dim=0)

                    if len(bboxes) < self.topk:
                        pad_length = self.topk - len(bboxes)
                        pad_bbox = torch.zeros((pad_length, 7)).to(
                            _bev3d_bbox.device
                        )
                        pad_scores = torch.zeros((pad_length,)).to(
                            _bev3d_bbox.device
                        )
                        pad_labels = (
                            torch.ones((pad_length,)).to(_bev3d_bbox.device)
                            * -99
                        )

                        bboxes = torch.cat((bboxes, pad_bbox))
                        scores = torch.cat((scores, pad_scores))
                        labels = torch.cat((labels, pad_labels))
                else:
                    bboxes = _bev3d_bbox
                    scores = _bev3d_score
                    labels = _bev3d_cls_id

                bev3d_bboxes_nms.append(bboxes)
                bev3d_score_nms.append(scores)
                bev3d_cls_id_nms.append(labels)

            bev3d_bboxes_nms = torch.stack(bev3d_bboxes_nms)
            bev3d_score_nms = torch.stack(bev3d_score_nms)
            bev3d_cls_id_nms = torch.stack(bev3d_cls_id_nms)

            # parsing the results
            pred["bev3d_ct"] = bev3d_bboxes_nms[:, :, :2]
            pred["bev3d_loc_z"] = bev3d_bboxes_nms[:, :, 2]
            pred["bev3d_dim"] = bev3d_bboxes_nms[:, :, 3:6]
            pred["bev3d_rot"] = bev3d_bboxes_nms[:, :, -1]
            pred["bev3d_score"] = bev3d_score_nms
            pred["bev3d_cls_id"] = bev3d_cls_id_nms

        return pred


@OBJECT_REGISTRY.register
class BEVDiscreteObjectDecoder(BEV3Decoder):
    """Decoder, convert model output to object results."""

    def __init__(self, **kwargs):
        super(BEVDiscreteObjectDecoder, self).__init__(**kwargs)

    def __call__(
        self,
        pred: Mapping,
        label: Dict = None,
    ) -> Dict:
        bev_discobj_hm = sigmoid_and_clip(pred["pred_bev_discobj_hm"])
        # max_pooling nms
        batch_size, _, height, width = bev_discobj_hm.shape
        kernel = self.max_pool_kernel
        pad = (kernel - 1) // 2
        max_bev_discobj_hm = F.max_pool2d(
            bev_discobj_hm, (kernel, kernel), stride=1, padding=(pad, pad)
        )
        bev_discobj_hm *= (max_bev_discobj_hm == bev_discobj_hm).float()
        pred["pred_bev_discobj_hm"] = bev_discobj_hm

        for key, out in pred.items():
            channel_size = out.shape[1]
            pred[key] = out.permute(0, 2, 3, 1).contiguous()
            pred[key] = pred[key].view(batch_size, -1, channel_size)

        # get the topk indices of heatmap
        bev_discobj_hm = pred.pop("pred_bev_discobj_hm")
        scores, classes = bev_discobj_hm.max(dim=-1)
        scores, topk_indices = scores.topk(k=self.topk, dim=1)
        pred["pred_bev_discobj_cls_id"] = classes

        # get the topk indices of regression items, e.g., dim, rot etc.
        for key, out in pred.items():
            pred[key] = torch.cat(
                [_out[_inds][None] for _out, _inds in zip(out, topk_indices)],
                dim=0,
            )

        # convert topk indices value to the coordinate
        pred["pred_bev_discobj_score"] = scores
        u, v = topk_indices % width, topk_indices // width  # bev: u=x, v=y
        bev_discobj_center = torch.cat(
            [u[:, :, None], v[:, :, None]], axis=-1
        )  # (u, v) = (x, y)

        # convert the bev center in bev coord to vcs coord
        vcs_center = self.coord_transform(
            bev_discobj_center,
            bev_size=(height, width),
            vcs_range=self.vcs_range,
        )
        pred["pred_bev_discobj_ct"] = vcs_center

        # convert the rot to rad value (sin / cos), range=[-pi, pi]
        bev_discobj_rot = pred.pop("pred_bev_discobj_rot")
        rot = torch.atan2(bev_discobj_rot[:, :, 1], bev_discobj_rot[:, :, 0])
        pred["pred_bev_discobj_rot"] = rot

        # clamp wh
        pred["pred_bev_discobj_wh"] = pred["pred_bev_discobj_wh"].clamp(
            0,
        )

        return pred
