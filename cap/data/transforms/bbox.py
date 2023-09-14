import numpy as np

__all__ = [
    "clip_bbox",
    "filter_bbox",
    "remap_bbox_label_by_area",
    "remap_bbox_label_by_clip_area_ratio",
]


def clip_bbox(bbox, img_roi, need_copy=False):
    """Clip bounding boxes according to roi area."""

    assert bbox.ndim == 2
    assert len(img_roi) == 4
    if not isinstance(img_roi, np.ndarray):
        img_roi = np.array(img_roi)
    if need_copy:
        bbox = bbox.copy()
    # clip border
    bbox[:, :2] = np.maximum(bbox[:, :2], img_roi[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], img_roi[2:4])
    return bbox


def filter_bbox(bbox, img_roi, allow_outside_center=True, min_edge_size=0.1):
    """Filter out bboxes.

    if its center outside img_roi or the maximum length
    of width and height is smaller than min_edge_size
    """
    assert bbox.ndim == 2
    assert len(img_roi) == 4
    roi = np.array(img_roi)
    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        centers = (bbox[:, :2] + bbox[:, 2:4]) / 2
        mask = np.logical_and(roi[:2] <= centers, centers < roi[2:]).all(
            axis=1
        )
    mask = np.logical_and(
        mask, (bbox[:, :2] + min_edge_size < bbox[:, 2:4]).all(axis=1)
    )
    bbox = bbox[mask]
    return bbox


def _get_bbox_area(bboxes):
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]
    return width * height


# TODO(alan):  should merge this two function into one.


def remap_bbox_label_by_area(bboxes, min_area=8, copy=False):
    """Map bbox label to be hard if its area < min_area."""

    area = _get_bbox_area(bboxes)
    to_be_hard_flag = np.logical_and(area < 8, bboxes[:, 4] > 0)
    if copy:
        bboxes = bboxes.copy()
    bboxes[to_be_hard_flag, 4] *= -1
    return bboxes


def remap_bbox_label_by_clip_area_ratio(
    before_clip_bboxes,
    after_clip_bboxes,
    valid_clip_area_ratio=0.5,
    copy=False,
):
    """
    Map bbox label to be hard.

    ifclip_bbox_area / bbox_are < valid_clip_are_ratio
    """

    old_area = _get_bbox_area(before_clip_bboxes)
    new_area = _get_bbox_area(after_clip_bboxes)
    clip_area_ratio = new_area / (old_area + 1e-14)
    to_be_hard_flag = np.logical_and(
        clip_area_ratio < valid_clip_area_ratio, after_clip_bboxes[:, 4] > 0
    )
    bboxes = after_clip_bboxes
    if copy:
        bboxes = bboxes.copy()
    bboxes[to_be_hard_flag, 4] *= -1
    return bboxes
