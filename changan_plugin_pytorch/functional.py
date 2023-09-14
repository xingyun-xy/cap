from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torchvision
import mmcv

from .qtensor import QTensor
from .utils.script_helper import get_script_subgraph

__all__ = [
    "max",
    "argmax",
    "round",
    "sort",
    "set_annotation",
    "get_output_annotation",
    "nms",
    "batched_nms",
    "box3d_iou_bev",
    "box3d_overlap_bev",
    "box3d_iou",
    "nms3d",
    "nms3d_normal",
    "multi_tensor_legacynadamex",
    "batched_nms_with_padding",
    "om_ogc",
    "om_confusion_matrix",
    "om_chamfer_distance",
]


def max(
    input: Union[Tensor, QTensor], dim: int = 1, keepdim: bool = True
) -> Union[Tuple[Tensor, Tensor], Tuple[QTensor, Tensor]]:
    """
    Please refer to torch.max for detailed info.

    Args:
        input (Union[Tensor, QTensor]): The input tensor in NCHW format.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether the output tensor has dim retained or not.

    Returns:
        Union[Tuple[Tensor, Tensor], Tuple[QTensor, Tensor]]:
            value: Max values in the shape of (N, 1 / None, H, W).
            idx: Index of max values in its own group
                in the shape of (N, 1 / None, H, W)
    """
    return input.max(dim, keepdim)


def argmax(
    input: Union[Tensor, QTensor], dim: int = 1, keepdim: bool = True
) -> Tensor:
    """
    Please refer to torch.argmax for detailed info.

    Args:
        input (Union[Tensor, QTensor]): The input tensor in NCHW format.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether the output tensor has dim retained or not.

    Returns:
        Tensor: Index of max values in its own group in the shape of
            (N, 1 / None, H, W)
    """
    return input.argmax(dim, keepdim)


@torch.jit.script
def _stable_sort(
    input: Tensor, dim: int = -1, descending: bool = False
) -> Tuple[Tensor, Tensor]:
    if input.numel() == 0 or input.dim() == 0:
        return torch.sort(input, dim, descending)
    return torch.ops.changan.sort(input, dim, descending)


@torch.jit.script
def sort(
    input: Tensor,
    dim: int = -1,
    descending: bool = False,
    stable: bool = False,
):
    """Please refer to torch.sort for detailed info.

    Args:
        input (Tensor): the input tensor.
        dim (int, optional): the dimension to sort along. Defaults to -1.
        descending (bool, optional): controls the sorting order (ascending or
        descending). Defaults to False.
        stable (bool, optional):  makes the sorting routine stable, which
        guarantees that the order of equivalent elements is preserved.
        Defaults to False.

    Returns:
        tuple: A namedtuple of (values, indices) is returned, where the values
        are the sorted values and indices are the indices of the elements in
        the original input tensor.
    """
    if stable:
        return _stable_sort(input, dim, descending)
    else:
        return torch.sort(input, dim, descending)


@torch.jit.script
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with
        IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been
        kept by NMS, sorted in decreasing order of scores
    """
    return torch.ops.changan.nms(boxes, scores, iou_threshold)


@torch.jit.script
def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float):
            discards all overlapping boxes with IoU > iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of
            the elements that have been kept by NMS, sorted
            in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, iou_threshold)
        # keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
        # _, keep = mmcv.ops.nms(boxes_for_nms, scores, iou_threshold, 
        #                 max_num=200000)
        # _, keep = mmcv.ops.nms(boxes_for_nms, scores, iou_threshold,
        #                        max_num=2000)
        return keep


@torch.jit.script
def round(data: Tensor) -> Tensor:
    """
    Round that round middle value away from zero.
    This behaviour is same as std::round.
    """
    return torch.ops.changan.round(data)


@torch.jit.script
def _set_annotation(data: Tensor, annotation: str) -> Tensor:
    """
    set tensor annotation internal
    """
    return data


def set_annotation(data: Tensor, annotation: str) -> Tensor:
    """
    set tensor annotation
    """
    r = _set_annotation(data, annotation)
    r.annotation = annotation
    return r


def get_output_annotation(script_model):
    """
    get output annotation from scripted model.
    the output of sctipt_model must be tensor or tuple of tensor or
    tuple of tuple of tensor.
    """
    assert isinstance(
        script_model, torch.jit.ScriptModule
    ), "please input script model use jit.trace or jit.script"
    anno_list = []
    passed_node = set()
    for out in script_model.graph.outputs():
        node = out.node()
        # tuple or function or forward
        if (
            node.kind() == "prim::TupleConstruct"
            or node.kind() == "prim::CallFunction"
            or node.kind() == "prim::CallMethod"
        ):
            anno_list.extend(
                _get_node_annotation(out, script_model, passed_node)
            )
        else:
            anno_list.append(None)
    return anno_list


def _get_node_annotation(node, model, passed_node):
    anno_list = []
    if node.debugName() in passed_node:
        return anno_list
    passed_node.add(node.debugName())
    node = node.node()
    if (
        node.kind() == "prim::TupleConstruct"
        or node.kind() == "prim::TupleUnpack"
    ):
        # tuple
        for i in node.inputs():
            anno_list.extend(_get_node_annotation(i, model, passed_node))
    elif node.kind() == "prim::CallMethod":
        # submodule forward
        subgraph = get_script_subgraph(model, node)
        if subgraph is not None:
            anno_list.extend(get_output_annotation(subgraph))
        else:
            anno_list.extend(_get_node_annotation(node, model, passed_node))
    elif node.kind() == "prim::CallFunction":
        # function of _set_annotation
        inputs = [i for i in node.inputs()]
        if (
            inputs[0].type().kind() == "FunctionType"
            and inputs[0].node().kind() == "prim::Constant"
            and inputs[0].node().hasAttribute("name")
            and inputs[0].node()["name"] == "_set_annotation"
        ):
            anno_list.append(inputs[2].node()["value"])
    else:
        anno_list.append(None)
    return anno_list


def box3d_iou_bev(boxes_a: torch.Tensor, boxes_b: torch.Tensor):
    """Calculate 3d boxes IoU based on bev overlap.

    Args:
        boxes_a (torch.Tensor): (N, 7) [x, y, z, dx, dy, dz, heading].
        boxes_b (torch.Tensor): (M, 7) [x, y, z, dx, dy, dz, heading].

    Returns:
        ans_iou: (N, M) torch.Tensor object.
    """
    ans_iou = torch.zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])),
        dtype=torch.float,
        device=boxes_a.device,
    )

    torch.ops.changan.box3d_iou_bev(boxes_a, boxes_b, ans_iou)

    return ans_iou


def box3d_overlap_bev(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """Calculate 3d boxes overlap under BEV view.

    This is a direct function call to CUDA overlap function.

    Args:
        boxes_a (torch.Tensor): (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b (torch.Tensor): (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_overlap: (N, M) torch.Tensor object, where
        ans_overlap[i, j] = overlap(boxes_a[i], boxes_b[j]).
    """
    ans_overlap = torch.zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])),
        dtype=torch.float,
        device=boxes_a.device,
    )

    torch.ops.changan.box3d_overlap_bev(boxes_a, boxes_b, ans_overlap)

    return ans_overlap


def box3d_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor):
    """Calculate 3d boxes IoU based on 3d volumetric overlap.
    Args:
        boxes_a: (torch.Tensor): (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (torch.Tensor): (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M) torch.Tensor object.
    """

    # transform back to pcdet's coordinate
    boxes_a = boxes_a[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes_a[:, -1] = -boxes_a[:, -1] - np.pi / 2
    boxes_b = boxes_b[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes_b[:, -1] = -boxes_b[:, -1] - np.pi / 2

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])),
        dtype=torch.float,
        device=boxes_a.device,
    )
    torch.ops.changan.box3d_overlap_bev(boxes_a, boxes_b, overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms3d(boxes: torch.Tensor, scores: torch.Tensor, thresh: float, **kwargs):
    """Perform 3d bounding box non-max suppression.

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading] 3d boxes.
        scores: (N) confidence of each box.
        thresh: nms overlap threshold.

    Return:
        Indices of boxes that survived the selection.
    """
    out = torch.ops.changan.nms3d(boxes, scores, thresh)
    return out


def nms3d_normal(boxes, scores, thresh, **kwargs):
    """Perform 3d bounding box non-max suppression. Boxes are not rotated.

    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading] 3d boxes.
        scores: (N) confidence of each box.
        thresh: nms overlap threshold.

    Return:
        Indices of boxes that survived the selection.
    """
    out = torch.ops.changan.nms3d_normal(boxes, scores, thresh)
    return out


def multi_tensor_legacynadamex(
    tensor_lists,
    step_list,
    lr,
    weight_decay,
    beta1,
    beta2,
    eps,
    schedule_decay,
    m_schedule,
    rescale_grad,
):
    """Perform fused LegacyNadamEx optimizer function.

    Args:
        tensor_lists: format is [param_list, grad_list, mean_list, var_list]
                      and each list must have same number of tensors.
                      Tensors will be modified same as origin optimizer.
        step_list: a list of step w.r.t each param.
        lr, weight_decay and all others is same as LegacyNadamEx optimizer.
    Return:
        m_schedule that has been updated. Must be saved for next step.
    """
    return torch.ops.changan.multi_tensor_legacynadamex(
        tensor_lists,
        step_list,
        lr,
        weight_decay,
        beta1,
        beta2,
        eps,
        schedule_decay,
        m_schedule,
        rescale_grad,
    )


def batched_nms_with_padding(
    boxes: Tensor,
    scores: Tensor,
    class_idxs: Optional[Tensor],
    iou_threshold: float,
    pre_top_n: int,
    post_top_n: int,
    legacy_bbox: bool,
    pad_mode: str = "pad_zero",
):
    """
    Batched Non-Maximum Supression.
    Output the index of preserved post_top_n boxes.
    Insufficient output will be padded to target number.

    Args:
        boxes (Tensor[N, box_num, 4]): Boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N, box_num]): Scores for each one of the boxes.
        class_idxs (Optional[Tensor[N, box_num]]):
            indices of the categories for each one of the boxes.
        iou_threshold (float):
            Discards all overlapping boxes with IoU > iou_threshold.
        pre_top_n (int): The top n bbox to apply nms on.
        post_top_n (int): The top n bbox to keep after nms.
        legacy_bbox (bool):
            Whether to add 1 when computing bounding box border.
        pad_mode (str, optional):
            The way to pad bbox to match the number of post_top_n.
            Defaults to "pad_zero".

    Returns:
        Tensor[N, box_num]: Preserved box index padded to target number.
    """
    if class_idxs is None:
        multi_class = False
        class_idxs = scores
    else:
        multi_class = True

    return torch.ops.changan.batched_nms(
        boxes,
        scores,
        class_idxs,
        iou_threshold,
        pre_top_n,
        post_top_n,
        multi_class,
        legacy_bbox,
        pad_mode,
    )


def om_ogc(
    pred_cls: Tensor,
    pred_prob: Tensor,
    pred_r: Tensor,
    pred_sin: Tensor,
    pred_cos: Tensor,
    pred_embedding: Tensor,
    pose_weights: Optional[Tensor],
    cls_num: int,
    cls_thr: float = 0.5,
    radius_l: int = 9,
    radius_t: int = 2,
    min_num: int = 1,
    cluster_thr: float = 0.95,
    merge: bool = False,
) -> Tensor:
    """
    OGC (Offset Growth Cluster) for Online Mapping Post Process.

    Notice: Generally, the channel C means set num, the default value is 2

    Args:
        pred_cls: tensor [C, H, W], class label
        pred_prob: tensor [C, H, W], class confidence
        pred_r: tensor [C, H, W], offset radius
        pred_sin: tensor [C, H, W], offset sin value
        pred_cos: tensor [C, H, W], offset cos value
        pred_embedding: tensor [C, H, W, D], embedding features for instance
        pose_weights: cpu 1D tensor, the weight of computing pose diversity
        cls_num: int, number of classes
        cls_thr: float, used to select valid pixels from pred_probs
        radius_l: int, the radius of longitudinal searching
        radius_t: int, the radius of transverse searching
        min_num: int, the minimum number of clustering points
        cluster_thr: float, the threshold of point similarly
        merge: bool, if merge clusters

    Returns:
        cluster_result: tensor [C, H, W], clsuter ids
    """
    if pose_weights is None:
        pose_weights = torch.tensor(
            [0.1] * cls_num, dtype=torch.float32, device="cpu"
        )
    elif isinstance(pose_weights, (list, tuple)):
        pose_weights = torch.tensor(
            pose_weights, dtype=torch.float32, device="cpu"
        )
    elif isinstance(pose_weights, dict):
        pose_weights = torch.tensor(
            list(pose_weights.values()), dtype=torch.float32, device="cpu"
        )
    elif not isinstance(pose_weights, Tensor):
        raise AssertionError(
            "Invalid ogc pose_weights:{}".format(pose_weights)
        )
    assert pose_weights.numel() == cls_num

    return torch.ops.changan.om_ogc(
        pred_cls.contiguous(),
        pred_prob.contiguous(),
        pred_r.contiguous(),
        pred_sin.contiguous(),
        pred_cos.contiguous(),
        pred_embedding.contiguous(),
        pose_weights.contiguous(),
        cls_num,
        cls_thr,
        radius_l,
        radius_t,
        min_num,
        cluster_thr,
        merge,
    )


def om_confusion_matrix(
    gt_cls: Tensor,
    pred_cls: Tensor,
    pred_prob: Tensor,
    cls_thr_list: Tensor,
    cls_num: int,
) -> Tensor:
    """
    Computing Confusion Matrix for Online Mapping Post Process.

    Args:
        gt_cls: tensor, ground truth class label
        pred_cls: tensor, prediction class label
        pred_prob: tensor, prediction class confidence
        cls_thr_list: tensor, confidence threshold list
        cls_num: int, number of classes

    Returns:
        result: tensor [len(cls_thr_list), cls_num, cls_num], confusion matrix
    """
    return torch.ops.changan.om_confusion_matrix(
        gt_cls.contiguous(),
        pred_cls.contiguous(),
        pred_prob.contiguous(),
        cls_thr_list.contiguous(),
        cls_num,
    )


def om_chamfer_distance(
    data_x: Tensor,
    data_y: Tensor,
    direction: str = "bi",
    dist_thresh: float = 10.0,
) -> float:
    """
    Computing Chamfer Distance for online mapping

    Args:
        data_x, cpu or gpu tensor, the first data
        data_y, cpu or gpu tensor, the second data
        direction, one of {"x_to_y", "y_to_x", "bi"}
        dist_thresh: default dist, if data is empty

    Returns:
        dist: float, chamfer distance value
    """
    if data_x.numel() == 0 or data_y.numel() == 0:
        if direction == "bi":
            return dist_thresh * 2.0
        else:
            return dist_thresh

    return torch.ops.changan.om_chamfer_distance(
        data_x.contiguous(),
        data_y.contiguous(),
        direction,
    )
