import copy
import json
import math
import os
import re
import warnings
from enum import Enum, unique

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cap.core.anno_ts_utils import (
    check_obj,
    check_parsing_ignore,
    draw_mask,
    drawcontour,
    get_image_shape,
    imread,
    imwrite,
    verify_image,
)
from cap.registry import OBJECT_REGISTRY, build_from_registry

__all__ = [
    "DenseBoxDetAnnoTs",
    "DefaultGenerateLabelMapAnnoTs",
    "DenseBoxSegAnnoTs",
    "KeyPointAnnoJsonTs",
    "KeyPointAnnoTs",
    "RearPlateKps4ToBBox",
    "ImageFailGenerateLabelMapAnnoTs",
]


@OBJECT_REGISTRY.register
class ImageFailGenerateLabelMapAnnoTs(object):
    """
    Generate mask from annotation for image fail parsing.

    Parameters
    ----------
    output_dir : str
        Output directory
    src_label : dict
        Source label ids.
    dst_label : dict
        Target label ids.
    colors : :py:class:`numpy.ndarray`
        Used in :py:func:`cv2.LUT`
    clsnames : iterable of str
        Class names.
    anno_to_contours_fn : callable
        How to convert annotation from polygons to mask.

        This function is called in the following ways:

        .. code-block:: python

            contours, value = anno_to_contours_fn(
                data, width, height, dst_label)
    reuse_prelabel : bool, optional
        Whether reuse prelabel, by default True.
    is_merge: bool, optional
        Whether merge label, by default True.
    merge_config: list, optional
        Config, by default None
    """

    def __init__(
        self,
        output_dir,
        src_label,
        dst_label,
        colors,
        clsnames,
        anno_to_contours_fn,
        reuse_prelabel=False,
        is_merge=False,
        merge_config=None,
    ):

        self.colors = colors[:, :, ::-1]
        self.anno_map = get_map(src_label, dst_label)

        self.labels = dst_label

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        draw_colorbar(colors, clsnames, os.path.split(self.output_dir)[0])

        self.area_count = copy.deepcopy(self.labels)
        for key in self.area_count.keys():
            self.area_count[key] = 0

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.reuse_prelabel = reuse_prelabel
        self.is_merge = is_merge
        self._anno_to_contours_fn = anno_to_contours_fn
        if is_merge:
            assert (
                merge_config is not None
            ), "merge_config should not be None, when is_merge is True"
            self.merge_config = merge_config

    def _anno_check(
        self,
        imgname,
        full_image_cts,
        partial_possible_cts,
        partial_certain_cts,
        ignore_cts,
    ):
        """
        Check annotation validity.

        Returns
        -------
        boolean
            true if check is passed.
        """
        result = True
        all_cts_num = (
            len(full_image_cts)
            + len(partial_possible_cts)
            + len(partial_certain_cts)
            + len(ignore_cts)
        )
        # check consistency in number of possible/certain
        if all_cts_num == 0:
            result = False
            warnings.warn(
                f"Invalid anno: there is no contours in {imgname}!"
            )  # noqa
        elif len(partial_possible_cts) != 0 and len(partial_certain_cts) == 0:
            result = False
            warnings.warn(
                f"Invalid anno: the possible/certain attrs are reversed in {imgname}!"  # noqa
            )
        elif all_cts_num == 1 and len(partial_possible_cts) == 1:
            result = False
            warnings.warn(
                f"Invalid anno: there is only one possible contour in {imgname}!"  # noqa
            )
        elif all_cts_num == 1 and len(partial_certain_cts) == 1:
            result = False
            warnings.warn(
                f"Invalid anno: there is only one partial certain contour in {imgname}!"  # noqa
            )
        elif len(full_image_cts) > 1:
            result = False
            warnings.warn(
                f"Invalid anno: there are more than one full image contour in {imgname}!"  # noqa
            )
        elif len(full_image_cts) == 1:
            ct = full_image_cts[0]
            if ct["attrs"]["type"] == "glare":
                result = False
                warnings.warn(
                    f"Invalid anno: full image glare is almost impossible in {imgname}!"  # noqa
                )
        return result

    def _draw_contours(self, mask, cts, width, height, imgname):
        draw_flag = True  # draw success or not
        cts_dict = {}  # ct type: cts
        full_image_cts = []
        partial_possible_cts = []
        partial_certain_cts = []
        ignore_cts = []
        # split cts by types
        for ct in cts:
            attrs = ct["attrs"]
            if attrs["ignore"] == "yes":
                ignore_cts.append(ct)
            elif attrs["photo"] == "yes":
                full_image_cts.append(ct)
            elif attrs["confidence"] == "possible":
                partial_possible_cts.append(ct)
            elif attrs["confidence"] == "certain":
                partial_certain_cts.append(ct)
            else:
                warnings.warn(
                    f"Invalid anno contour is found in {imgname}: {attrs}!"
                )  # noqa
        # anno check
        if not self._anno_check(
            imgname,
            full_image_cts,
            partial_possible_cts,
            partial_certain_cts,
            ignore_cts,
        ):
            draw_flag = False
            return mask, draw_flag
        # the order is important
        cts_dict[ImageFailCtsType.FULL_IMAGE] = full_image_cts
        cts_dict[ImageFailCtsType.PARTIAL_POSSIBLE] = partial_possible_cts
        cts_dict[ImageFailCtsType.PARTIAL_CERTAIN] = partial_certain_cts
        cts_dict[ImageFailCtsType.IGNORE] = ignore_cts
        for ct_type, cts in cts_dict.items():
            contours = []
            values = []
            if len(cts) > 0:
                for ct in cts:
                    ct_pts, value = self._anno_to_contours_fn(
                        ct_type, ct, width, height, self.labels, imgname
                    )
                    contours.append(ct_pts)
                    values.append(value)
                assert len(contours) == len(values)
                for value, contour in zip(values, contours):
                    mask = drawcontour(contour, value, mask)
        return mask, draw_flag

    def _merge_labelmap(
        self,
        label,
        merge_id_src,
        merge_label,
        merge_id_dst,
        merge_iou_thresh,
        merge_pixel_thresh,
    ):
        merge_label = np.array(merge_label == merge_id_dst) * (label != 255)
        h, w = label.shape
        label_merge = copy.deepcopy(label)
        _, contours, hierarchy = cv2.findContours(
            merge_label.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            # M = drawcontour(contour, h, w)
            M = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(M, [contour], -1, (1), -1)
            if merge_pixel_thresh is not None:
                if M.sum() < merge_pixel_thresh:
                    continue
            if merge_iou_thresh is not None:
                intersection = label[M == 1] == merge_id_src
                ratio = intersection.sum() / (M.sum() + 0.0)
                if ratio > merge_iou_thresh:
                    label_merge[M == 1] = merge_id_src
            else:
                label_merge[M == 1] = merge_id_src
        label_merge[label == 255] = 255
        return label_merge

    def __call__(self, item):
        if item is None:
            return None
        image_dir, anno = item
        image_path = os.path.join(image_dir, anno["image_key"])
        image_path = os.path.abspath(image_path)
        if "parsing" not in anno:
            warnings.warn(
                "WARNING! no paring area: %s, ignore." % anno["image_key"]
            )  # noqa
            return None

        image_path = _default_check_render_image(image_path)
        name = os.path.basename(image_path)
        anno["image_key"] = name
        anno["image_url"] = image_path

        imgname = name.replace("." + name.split(".")[-1], "")

        im = imread(image_path)
        if im is None:
            warnings.warn("WARNING! no image: %s, ignore." % image_path)
            return None
        predict_label = image_path.replace(
            "." + image_path.split(".")[-1], "_label.png"
        )
        if self.reuse_prelabel and os.path.isfile(predict_label):
            label = self.anno_map[imread(predict_label)]
        else:
            label = np.zeros(im.shape[0:2], dtype=np.uint8)

        cts = anno["parsing"]
        width, height = anno["width"], anno["height"]
        label, draw_flag = self._draw_contours(
            label, cts, width, height, imgname
        )
        if not draw_flag:
            return None

        if self.is_merge:
            assert "merge_label_url" in anno, "Not exist merge label."
            merge_label = imread(anno["merge_label_url"])
            for merge_info in self.merge_config:
                label = self._merge_labelmap(
                    label,
                    merge_info["merge_id_src"],
                    merge_label,
                    merge_info["merge_id_dst"],
                    merge_info["merge_iou_thresh"],
                    merge_info["merge_pixel_thresh"],
                )
        label_path = self.output_dir + "/" + imgname + "_label.png"
        imwrite(label_path, label)
        label_fusion_path = self.output_dir + "/" + imgname + "_fusion.jpg"
        imwrite(label_fusion_path, draw_mask(im, label, self.colors))

        anno["label_path"] = label_path
        return image_path, anno


@OBJECT_REGISTRY.register
class RearPlateKps4ToBBox(object):
    def __init__(self, classname):
        self.classname = classname
        self._bbox_occlusion = [
            "full_visible",
            "occluded",
            "heavily_occluded",
            "invisible",
        ]

    def __call__(self, item):
        if len(item) == 1 and item[0] is not None:
            image_dir, anno = item[0]
        elif len(item) == 2:
            image_dir, anno = item
        else:
            return None

        for obj in anno.get(self.classname, []):
            points = np.array(obj["data"]).reshape(4, 2)
            xmin = points[:, 0].min()
            ymin = points[:, 1].min()
            xmax = points[:, 0].max()
            ymax = points[:, 1].max()

            num_occllusion_points = 0
            bbox_ignore = "no"
            for point_attr in obj["point_attrs"]:
                if point_attr["point_label"]["occlusion"] != "full_visible":
                    num_occllusion_points += 1
                if point_attr["point_label"]["ignore"] == "yes":
                    bbox_ignore = "yes"
            num_occllusion_points = min(
                len(self._bbox_occlusion) - 1, num_occllusion_points
            )
            bbox_occlusion = self._bbox_occlusion[num_occllusion_points]

            obj["data"] = [xmin, ymin, xmax, ymax]
            obj["struct_type"] = "rect"
            obj["label_type"] = "boxes"
            bbox_attrs = {
                "ignore": bbox_ignore,
                "occlusion": bbox_occlusion,
            }
            obj["attrs"] = bbox_attrs

        return ((image_dir, anno),)


def default_anno_to_contours_fn(data, width, height, labels):
    """
    Transfrom the annotation data to contours.

    Parameters
    ----------
    data: dict
        the annotation label.
    width: int
        the image width.
    height: int
        the image height.
    labels: dict
        the label map.
    """
    pts = data["data"]
    if None in pts:
        return None
    ct_pts = []
    for pt in pts:
        ct_pts.append(
            (
                max(min(int(float(pt[0])), width - 1), 0),
                max(min(int(float(pt[1])), height - 1), 0),
            )
        )
    ct_pts = np.array(ct_pts)

    value = 0
    value = labels[data["attrs"]["type"]]
    if "ignore" in data["attrs"] and data["attrs"]["ignore"] == "yes":
        value = 255
    elif (
        "double_line" in data["attrs"]
        and data["attrs"]["double_line"] == "yes"
    ):  # noqa
        value = labels["double_line"]
    assert value != -1, "{}, {}".format(
        data["attrs"]["type"], data["attrs"]["ignore"]
    )  # noqa
    return ct_pts, value


def _is_valid_jpg(jpg_file):
    with open(jpg_file, "rb") as f:
        f.seek(-10, 2)
        buf = f.read()
        return b"\xff\xd9" in buf


def _check_image_completeness(image_path):
    """Check whether an image is broken or not."""
    if image_path.endswith("jpeg") or image_path.endswith("jpg"):
        return _is_valid_jpg(image_path)

    elif image_path.endswith("png"):
        return True
    else:
        raise NotImplementedError(
            "Check image completeness temporary support only .jpg and .png"
        )


def _get_box_10_points(bbox):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    width = x2 - x1
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    points_data = [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
        [cx, cy],
        [cx + (width + height) / 2.0, cy],
        [cx + (width * height) ** 0.5, cy],
        [cx + max(width, height), cy],
        [cx + max(2 * width, height), cy],
        [cx + max(width, 2 * height), cy],
    ]

    return points_data


@OBJECT_REGISTRY.register
class DenseBoxDetAnnoTs(object):
    """Default annotation transformer for object detection.

    Packed in the densebox image record format.

    Args:
        anno_config : dict
            Configure
        root_dir : str
            Image root
    """

    def __init__(self, anno_config, root_dir, verbose=True, skip_invalid=True):
        self.verbose = verbose
        self.root_dir = root_dir
        self.anno_config = anno_config
        self.skip_invalid = skip_invalid

    def __call__(self, *item):
        if len(item) == 1 and item[0] is not None:
            image_dir, anno = item[0]
        elif len(item) == 2:
            image_dir, anno = item
        else:
            return None

        # image_url = anno["image_url"]
        image_url = os.path.join(image_dir, anno["image_key"])
        if not os.path.exists(image_url):
            if self.skip_invalid:
                if self.verbose:
                    warnings.warn(
                        "WARNING: skip invalid image: %s" % (image_url)
                    )
                return None
            else:
                raise RuntimeError("No such image: %s" % (image_url))

        if not _check_image_completeness(image_url):
            if self.verbose:
                warnings.warn(
                    "WARNING: skip premature end image: %s" % (image_url)
                )  # noqa
            return None

        instances = []
        ignore_regions = []
        for obj in anno.get(self.anno_config["base_classname"], []):
            x1, y1, x2, y2 = map(float, obj["data"])
            height = y2 - y1
            width = x2 - x1
            if height <= 0 or width <= 0:
                continue
            points_data = _get_box_10_points(map(float, obj["data"]))
            matched = False
            for class_mapper in self.anno_config["class_mappers"]:
                if check_obj(obj, class_mapper.get("match_condiction")):
                    matched = True
                    class_id = class_mapper["id"]
                    if class_id is None:
                        upsample = class_mapper.get("upsample", 0)
                        if upsample == 0:
                            break
                        assert (
                            upsample < 1
                        ), "ignore supports downsample only! Plese set upsample value < 1"  # noqa
                        if upsample > np.random.rand():
                            continue  # keep upsample of this match_condition and continue mapping  # noqa
                        else:
                            break  # ignore 1-upsample
                    if check_obj(obj, class_mapper.get("ignore_condiction")):
                        ignore_region = {
                            "left_top": points_data[0],
                            "right_bottom": points_data[2],
                            "class_id": [class_id],
                        }
                        ignore_regions.append(ignore_region)
                    else:
                        if check_obj(obj, class_mapper.get("hard_condiction")):
                            is_hard = True
                        else:
                            is_hard = False
                        instance = {
                            "points_data": points_data,
                            "class_id": [class_id],
                            "attribute": [],
                            "is_hard": [int(is_hard)],
                        }
                        upsample = class_mapper.get("upsample", 1)
                        upsample_int = int(upsample)
                        upsample = upsample_int + int(
                            (upsample - upsample_int) > np.random.rand()
                        )
                        instances.extend([instance] * upsample)
                    if not self.anno_config.get("allow_multi_match", False):
                        break
            if not matched and self.verbose:
                warnings.warn("WARNING: not matched obj: %s" % json.dumps(obj))

        if self.anno_config.get("remove_empty_images", False) and not len(
            instances
        ):  # noqa
            if self.verbose:
                warnings.warn("WARNING: skip no data image: %s" % (image_url))
            return None
        np.random.shuffle(instances)

        img = cv2.imread(image_url, cv2.IMREAD_UNCHANGED)

        if img is None:
            if self.verbose:
                warnings.warn("WARNING: skip invalid image: %s" % (image_url))
            return None
        img_url = os.path.relpath(image_url, self.root_dir)
        if self.anno_config.get("remove_zh_image_path", False) and re.findall(
            "[\u4e00-\u9fa5]", img_url
        ):  # noqa
            if self.verbose:
                warnings.warn(
                    "WARNING, deprecated params: skip zh image path: %s"
                    % (image_url)
                )  # noqa
            return None
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_c = img.shape[2] if len(img.shape) == 3 else 1

        if self.anno_config.get("default_ignore_full_image", False):
            class_ids = set(
                map(
                    lambda class_mapper: class_mapper["id"],
                    self.anno_config["class_mappers"],
                )
            )
            for class_id in range(1, self.anno_config["num_classes"] + 1):
                if class_id not in class_ids:
                    ignore_region = {
                        "left_top": (0, 0),
                        "right_bottom": (img_w, img_h),
                        "class_id": [class_id],
                    }
                    ignore_regions.append(ignore_region)

        anno_dict = {
            "img_url": img_url,
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "instances": instances,
            "ignore_regions": ignore_regions,
        }
        return anno_dict


def get_map(src_label, dst_label):
    """
    Get the mapping from src_label to dst_label.

    Parameters
    ----------
    src_label: dict
        the source label.
    dst_label: dict
        the destination label.
    """
    assert len(src_label.keys()) == len(dst_label.keys()), "{},{}".format(
        len(src_label.keys()), len(dst_label.keys())
    )
    anno_map = 256 * np.ones(256)
    for i in src_label.keys():
        if anno_map[src_label[i]] < 256:
            assert anno_map[src_label[i]] == dst_label[i], (
                "Please check the value of key {} (value: {})."
                "The value is conflict with other key(s). {} vs {}".format(
                    i, src_label[i], anno_map[src_label[i]], dst_label[i]
                )
            )
        anno_map[src_label[i]] = dst_label[i]
    return anno_map


def draw_colorbar(colors, clsnames, outpath):
    """Draw color bar for segmentation task.

    Args:
        colors: list of int
            RGB color of each segmentation class.
        clsnames: list of str
            segmentation class names.
        outpath: str
            Color bar output path.
    """

    def putText(im, str, position, scale):
        img_PIL = Image.fromarray(im[:, :, ::-1])
        ImageDraw.Draw(img_PIL).text(position, str, fill=(255, 255, 255))
        return np.asarray(img_PIL)[:, :, ::-1]

    num_cls = len(clsnames)

    max_c = 0
    for name in clsnames:
        max_c = max(max_c, len(name))

    length = int(720 / int(num_cls))
    bar = np.zeros((720, int(0.4 * length) * (max_c + 2), 3))
    for i in range(int(num_cls)):
        bar[i * length : (i + 1) * length, :, 0] = colors[i, :, 2]
        bar[i * length : (i + 1) * length, :, 1] = colors[i, :, 1]
        bar[i * length : (i + 1) * length, :, 2] = colors[i, :, 0]
    bar[int(num_cls) * length :, :, 0] = colors[int(num_cls) - 1, :, 2]
    bar[int(num_cls) * length :, :, 1] = colors[int(num_cls) - 1, :, 1]
    bar[int(num_cls) * length :, :, 2] = colors[int(num_cls) - 1, :, 0]

    bar_show = bar.astype("uint8")
    for i in range(int(num_cls)):
        str = "%02d %s" % (i, clsnames[i])
        bar_show = putText(
            bar_show,
            str,
            (int(0.2 * length), int(i * length)),
            int(0.6 * length),
        )

    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    cv2.imwrite(
        os.path.join(outpath, "colorbar_{}.png".format(num_cls)),
        bar_show.astype("uint8"),
    )


def _default_check_render_image(image_path):
    """Check given the path of an image is a render image.

    If is a render image, return the path of source image.

    Args:
        image_path: str
            the absolute path of an image.
    """
    image_path = os.path.abspath(image_path)
    assert (
        image_path.endswith("jpg")
        or image_path.endswith("jpeg")
        or image_path.endswith("png")
    ), ("invalid ext for image %s, only allow jpg, jpeg, png" % image_path)

    if "_render" in image_path:
        assert image_path.endswith("_render.jpg")
        if os.path.isfile(image_path.replace("_render.jpg", ".jpeg")):
            image_path = image_path.replace("_render.jpg", ".jpeg")
        elif os.path.isfile(image_path.replace("_render.jpg", ".jpg")):
            image_path = image_path.replace("_render.jpg", ".jpg")
        else:
            raise RuntimeError(
                "error! cannot find raw image for rendered image %s!"
                % (image_path)
            )  # noqa
    return image_path


@OBJECT_REGISTRY.register
class DefaultGenerateLabelMapAnnoTs(object):
    """Generate mask from annotation.

    Args:
        output_dir : str
            Output directory
        src_label : dict
            Source label ids.
        dst_label : dict
            Target label ids.
        colors : :py:class:`numpy.ndarray`
            Used in :py:func:`cv2.LUT`
        clsnames : iterable of str
            Class names.
        anno_to_contours_fn : callable
            How to convert annotation from polygons to mask.

            This function is called in the following ways:

            .. code-block:: python

                contours, value = anno_to_contours_fn(
                    data, width, height, dst_label)
        reuse_prelabel : bool, optional
            Whether reuse prelabel, by default True.
        is_merge: bool, optional
            Whether merge label, by default False.
        merge_config: list, optional
            Config, by default None
    """

    def __init__(
        self,
        output_dir,
        src_label,
        dst_label,
        colors,
        clsnames,
        anno_to_contours_fn,
        dst_label_map=None,
        reuse_prelabel=True,
        is_merge=False,
        merge_config=None,
        check_parsing_ignore=False,
    ):

        self.colors = colors[:, :, ::-1]
        self.anno_map = get_map(src_label, dst_label)

        self.labels = dst_label
        self.dst_label_map = dst_label_map

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        draw_colorbar(colors, clsnames, os.path.split(self.output_dir)[0])

        self.area_count = copy.deepcopy(self.labels)
        for key in self.area_count.keys():
            self.area_count[key] = 0

        self.output_dir = os.path.abspath(output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.reuse_prelabel = reuse_prelabel
        self.is_merge = is_merge
        self.check_parsing_ignore = check_parsing_ignore
        self._anno_to_contours_fn = anno_to_contours_fn
        if is_merge:
            assert (
                merge_config is not None
            ), "merge_config should not be None, when is_merge is True"
            self.merge_config = merge_config

    def _draw_contours(self, mask, cts, width, height):
        contours = []
        values = []
        for c in cts:
            if self.dst_label_map is None:
                ct_pts, value = self._anno_to_contours_fn(
                    c, width, height, self.labels
                )
            else:
                ct_pts, value = self._anno_to_contours_fn(
                    c, width, height, self.dst_label_map
                )
            contours.append(ct_pts)
            values.append(value)

        assert len(contours) == len(values)
        for value, contour in zip(values, contours):
            mask = drawcontour(contour, value, mask)

        return mask

    def _merge_labelmap(
        self,
        label,
        merge_id_src,
        merge_label,
        merge_id_dst,
        merge_iou_thresh,
        merge_pixel_thresh,
    ):
        merge_label = np.array(merge_label == merge_id_dst) * (label != 255)
        h, w = label.shape
        label_merge = copy.deepcopy(label)
        _, contours, hierarchy = cv2.findContours(
            merge_label.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            # M = drawcontour(contour, h, w)
            M = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(M, [contour], -1, (1), -1)
            if merge_pixel_thresh is not None:
                if M.sum() < merge_pixel_thresh:
                    continue
            if merge_iou_thresh is not None:
                intersection = label[M == 1] == merge_id_src
                ratio = intersection.sum() / (M.sum() + 0.0)
                if ratio > merge_iou_thresh:
                    label_merge[M == 1] = merge_id_src
            else:
                label_merge[M == 1] = merge_id_src
        label_merge[label == 255] = 255
        return label_merge

    def __call__(self, item):
        if item is None:
            return None
        image_dir, anno = item
        image_path = os.path.join(image_dir, anno["image_key"])
        image_path = os.path.abspath(image_path)
        if "parsing" not in anno:
            warnings.warn(
                "WARNING! no paring area: %s, ignore." % anno["image_key"]
            )  # noqa
            return None

        image_path = _default_check_render_image(image_path)
        name = os.path.basename(image_path)
        anno["image_key"] = name
        anno["image_url"] = image_path

        print("image_key",name)

        imgname = name.replace("." + name.split(".")[-1], "")

        im = imread(image_path)
        if im is None:
            warnings.warn("WARNING! no image: %s, ignore." % image_path)
            return None
        if self.reuse_prelabel:
            if "predict_label" in anno:
                predict_label = anno["predict_label"]
            else:
                predict_label = image_path.replace(
                    "." + image_path.split(".")[-1], "_label.png"
                )
            assert os.path.isfile(
                predict_label
            ), f"Not exist predict label path: {predict_label}"
            label = self.anno_map[imread(predict_label)]
        else:
            label = np.zeros(im.shape[0:2], dtype=np.uint8)

        cts = anno["parsing"]
        width, height = anno["width"], anno["height"]
        label = self._draw_contours(label, cts, width, height)

        if self.is_merge:
            assert "merge_label_url" in anno, "Not exist merge label."
            merge_label = imread(anno["merge_label_url"])
            for merge_info in self.merge_config:
                label = self._merge_labelmap(
                    label,
                    merge_info["merge_id_src"],
                    merge_label,
                    merge_info["merge_id_dst"],
                    merge_info["merge_iou_thresh"],
                    merge_info["merge_pixel_thresh"],
                )
        label_path = self.output_dir + "/" + imgname + "_label.png"
        imwrite(label_path, label)
        label_fusion_path = self.output_dir + "/" + imgname + "_fusion.jpg"
        imwrite(label_fusion_path, draw_mask(im, label, self.colors))
        if self.check_parsing_ignore and not check_parsing_ignore(label_path):
            if os.path.isfile(label_path):
                os.remove(label_path)
            if os.path.isfile(label_fusion_path):
                os.remove(label_fusion_path)
            return None

        anno["label_path"] = label_path
        return image_path, anno


class _default_get_label_path_transformer(object):
    """Given the path of an image.

    Return the path of its corresponding label map file.
    `Label_root_path` is specified in this file.

    Args:
        label_root_path: str
            the label map root path; if None, use image root path.
    """

    def __init__(self, label_root_path=None):
        if label_root_path is not None:
            self.label_root_path = os.path.abspath(label_root_path)
        else:
            self.label_root_path = label_root_path

    def __call__(self, image_path):
        image_path = os.path.expanduser(image_path)
        assert (
            image_path.endswith("jpg")
            or image_path.endswith("jpeg")
            or image_path.endswith("png")
        ), ("invalid ext for image %s, only allow jpg, jpeg, png" % image_path)

        if self.label_root_path is not None:
            image_name = os.path.basename(image_path)
            label_name = os.path.splitext(image_name)[0] + "_label.png"
            label_path = os.path.join(self.label_root_path, label_name)
        else:
            label_path = image_path[: image_path.rindex(".")] + "_label.png"
        return label_path


@OBJECT_REGISTRY.register
class DenseBoxSegAnnoTs(object):
    """Legacy transformer.

    That transform segmentation annotation to densebox detection annotation.

    Args:
        class_ids: list of int
            Class ids for instances in segmentation annotation
        get_label_path_fn: callable or None
            The way to get label path. This function is called
            on the following way

            .. code-block:: none

                label_path = get_label_path_fn(image_path)
        verify_image: bool
            Whether to verify image.
    """

    def __init__(
        self,
        class_ids,
        get_label_path_fn=_default_get_label_path_transformer(),  # noqa
        verify_image=True,
        verify_label=True,
    ):

        assert isinstance(class_ids, list), "expect list, but get %s" % str(
            type(class_ids)
        )
        assert callable(
            get_label_path_fn
        ), "get_label_path_fn should be callable"
        for k, v in enumerate(class_ids):
            assert (
                isinstance(v, int) and v >= 0
            ), "idx should >= 0, but get %s at key %s" % (str(type(v)), k)
        self.class_ids = class_ids
        self.get_label_path_fn = get_label_path_fn
        self.verify_image = verify_image
        self.verify_label = verify_label

    def __call__(self, *item):
        if item[0] is None:
            return None
        if len(item) == 2:
            _, anno = item
        elif len(item) == 1:
            _, anno = item[0]
        else:
            raise RuntimeError

        img_path = anno["image_url"]
        label_path = anno.get("label_path", None)

        img_path = _default_check_render_image(img_path)

        if label_path is None:
            label_path = self.get_label_path_fn(img_path)

        if self.verify_image and not verify_image(img_path):
            return None
        if self.verify_label and not verify_image(label_path):
            return None
        img_h, img_w, img_c = get_image_shape(cv2.imread(img_path))
        l, t, r, b = (0, 0, img_w, img_h)
        cx, cy = (round(0.5 * img_w, 2), round(0.5 * img_h, 2))
        inst = {
            "points_data": [[l, t], [r, t], [r, b], [l, b], [cx, cy]],
            "class_id": self.class_ids,
            "attribute": [],
            "is_hard": [0],
            "is_point_hard": [0] * 5,
        }
        # no ignore region
        anno_dict = {
            "instances": inst,
            "ignore_regions": None,
            "img_url": img_path,
            "parsing_map_urls": label_path,
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "idx": -1,
            "img_attribute": None,
        }
        return anno_dict


@OBJECT_REGISTRY.register
class KeyPointAnnoJsonTs(object):
    """
    Transform keypoint annotation.

    To generic keypoint and detection annotation.

    Parameters
    ----------
    num_kps: int
        the number of kps in anno
    class_name: str
        classname for kps
    """

    def __init__(self, num_kps, conf_ignore_list, class_name="vehicle"):
        self.num_kps = num_kps
        self.num_box = 0
        self.conf_ignore_list = conf_ignore_list
        self.class_name = class_name

    def __call__(self, item):
        """
        Parameters.

        ----------
        anno: dict
            Annotation in CHan image data annotation.

        Returns
        -------
        trav_result : list
            Bounding box and keypoint information.
        img_abs_path : str
            Image absolute path.

        trav_result : list

            When no conf, returns

            .. code-block:: python
                [
                    bbox_type, bbox_clur, bbox_occ, bbox_ignore, bbox_xmin,
                    bbox_ymin, bbox_xmax, bbox_ymax, kp_back_xmin, kp_back_ymin,
                    kp_front_xmax, kp_front_ymax, occ_back, occ_front
                ]

            When conf exists, returns

            .. code-block:: python
                [
                    bbox_type, bbox_clur, bbox_occ, bbox_ignore, bbox_xmin,
                    bbox_ymin, bbox_xmax, bbox_ymax, kp_back_xmin, kp_back_ymin,
                    kp_front_xmax, kp_front_ymax, occ_back, occ_front, conf_back,
                    conf_front
                ]
        """  # noqa
        (img_dir, anno) = item[0]

        if (
            self.class_name not in anno.keys()
            or "p_WheelKeyPoints_" + str(self.num_kps) not in anno.keys()
        ):
            return None
        if "belong_to" not in anno.keys():
            return None

        # record image_key
        image_url = os.path.join(img_dir, anno["image_key"])
        img_abs_path = os.path.abspath(image_url)

        # create bbox dict
        data_bbox = anno[self.class_name]
        bbox_dict = {}
        for bbox_ele in data_bbox:
            if "id" not in bbox_ele.keys():
                continue
            bbox_dict[int(bbox_ele["id"])] = bbox_ele

        # parse matching dict {kps_id: bbox_id}
        matching = anno["belong_to"]
        matching_dict = {}
        for matching_ele in matching:
            matching_list = matching_ele.strip().split(":")

            bbox_id = int(matching_list[0].strip().split("|")[-1])
            kps_id = int(matching_list[1].strip().split("|")[-1])

            matching_dict[kps_id] = bbox_id

        # traverse
        trav_result = []

        data_wheel_kps = anno["p_WheelKeyPoints_" + str(self.num_kps)]
        has_conf = isinstance(data_wheel_kps[0]["point_attrs"][0], dict)
        for kps_ele in data_wheel_kps:
            temp_result = []
            if "id" not in kps_ele.keys():
                continue

            kps_id = kps_ele["id"]
            if kps_id not in matching_dict.keys():
                continue

            if "data" not in kps_ele or len(kps_ele["data"]) < self.num_kps:
                continue

            matched_bbox = bbox_dict[matching_dict[kps_id]]
            matched_bbox_attrs = matched_bbox["attrs"]

            temp_result.append(matched_bbox_attrs["type"])
            if "blur" in matched_bbox_attrs.keys():
                temp_result.append(matched_bbox_attrs["blur"])
            else:
                temp_result.append("Normal")
            temp_result.append(
                matched_bbox_attrs.get("occlusion", "fullvisible")
            )
            temp_result.append(matched_bbox_attrs.get("ignore", "no"))
            temp_result.append(str(matched_bbox["data"][0]))
            temp_result.append(str(matched_bbox["data"][1]))
            temp_result.append(str(matched_bbox["data"][2]))
            temp_result.append(str(matched_bbox["data"][3]))
            for kps_i in range(self.num_kps):
                temp_result.append(str(kps_ele["data"][kps_i][0]))
                temp_result.append(str(kps_ele["data"][kps_i][1]))
            # if not has_conf:
            if isinstance(kps_ele["point_attrs"][0], str):
                for kps_i in range(self.num_kps):
                    temp_result.append(kps_ele["point_attrs"][kps_i])
            # else:
            elif isinstance(kps_ele["point_attrs"][0], dict):
                kps_attr = kps_ele["point_attrs"]
                for kps_i in range(self.num_kps):
                    # occlusion
                    temp_result.append(
                        kps_attr[kps_i]["point_label"]["occlusion"]
                    )
                for kps_i in range(self.num_kps):
                    # confidence
                    temp_conf = kps_attr[kps_i]["point_label"][
                        "Corner_confidence"
                    ]
                    if temp_conf in self.conf_ignore_list:
                        continue
                    temp_result.append(temp_conf)
            else:
                raise TypeError('data "point_attrs" type error')

            # update valid box number
            self.num_box += 1
            trav_result.append(temp_result)

        return (
            img_abs_path,
            trav_result,
            anno["width"],
            anno["height"],
            has_conf,
        )


@OBJECT_REGISTRY.register
class KeyPointAnnoTs(object):
    """
    Key point annotation transformer.

    Parameters
    ----------
    num_kps: int
        the number of kps in anno
    """

    def __init__(
        self,
        num_kps,
        valid_types,
        min_width,
        min_height,
        kps_occ_id_dict,
        occ_ignore_list,
        obj_ele_num_2kps,
        obj_ele_num_3kps,
        obj_ele_num_xkps,
        skip_invalid=True,
    ):

        # assert num_kps in [2, 3], 'num_kps should be 2 or 3.'
        self.valid_types = valid_types
        self.min_width = min_width
        self.min_height = min_height
        self.kps_occ_id_dict = kps_occ_id_dict
        self.occ_ignore_list = occ_ignore_list

        self.num_kps = num_kps
        self.obj_ele_num_2kps = obj_ele_num_2kps
        self.obj_ele_num_3kps = obj_ele_num_3kps
        self.obj_ele_num_xkps = obj_ele_num_xkps

        self.ret_num = 0
        self.num_valid_bbox = 0
        self.kps_count_dict = {}
        self.skip_invalid = skip_invalid

    def check_occ(self, *occ):
        for data in occ:
            if data not in self.kps_occ_id_dict:
                return False
        return True

    def __call__(
        self,
        img_abs_path,
        trav_result=None,
        width=None,
        height=None,
        has_conf=False,
    ):
        if img_abs_path is None:
            return None
        else:
            img_abs_path, trav_result, width, height, has_conf = img_abs_path

        elem_per_inst = self.obj_ele_num_xkps
        if has_conf:
            elem_per_inst += self.num_kps

        num_obj = len(trav_result)
        if num_obj == 0:
            return None

        if elem_per_inst != len(trav_result[0]):
            print("[!] Error anno data ")
            if self.skip_invalid:
                return None
            else:
                raise RuntimeError

        # generate label, traverse valid obj
        boxes, gt_classes, keypoints = [], [], []
        for obj_data in trav_result:
            cls_cur, blur, occ, ignore = obj_data[:4]

            if cls_cur not in self.valid_types:
                continue

            if blur != "Normal" and blur != "no":
                continue

            # check object occ
            if occ in self.occ_ignore_list:
                continue

            # check 2d bbox dimension
            x1, y1, x2, y2 = map(float, obj_data[4 : 4 + 4])
            bbox_width, bbox_height = x2 - x1, y2 - y1
            if bbox_width < self.min_width or bbox_height < self.min_height:
                continue

            # assign box, category and keypoints
            boxes.append([x1, y1, x2, y2])
            gt_classes.append(1)

            keypoints_per_obj = []
            for kps_i in range(self.num_kps):
                # 1 + 3 + 4 + num_kps * 2 + num_kps
                occ_kp_xkps = obj_data[8 + self.num_kps * 2 + kps_i]

                if not self.check_occ(occ_kp_xkps):
                    print("[!] Error anno data %s" % img_abs_path)
                    if self.skip_invalid:
                        return None
                    else:
                        raise RuntimeError

                keypoints_per_obj.extend(
                    [
                        float(obj_data[8 + 2 * kps_i]),
                        float(obj_data[8 + 2 * kps_i + 1]),
                        self.kps_occ_id_dict[occ_kp_xkps],
                    ]
                )

                if occ_kp_xkps not in self.kps_count_dict:
                    self.kps_count_dict[occ_kp_xkps] = 0
                self.kps_count_dict[occ_kp_xkps] += 1
            keypoints.append(keypoints_per_obj)
        # no valid obj
        if len(boxes) <= 0 or len(gt_classes) <= 0 or len(keypoints) <= 0:
            return None

        self.num_valid_bbox += len(boxes)

        # assign label
        roi_rec = {
            "image": img_abs_path,
            "height": height,
            "width": width,
            "boxes": boxes,
            "gt_classes": gt_classes,
            "keypoints": keypoints,
        }
        self.ret_num += 1
        return roi_rec


@OBJECT_REGISTRY.register
class KeyPointEmptyTs(object):
    """
    Empty key point transform.

    Parameters
    ----------
    num_kps: int
        the number of kps in anno
    config: dict
        config file
    """

    def __init__(self):
        pass

    def __call__(self, roi_rec):
        keypoints = np.array(roi_rec["keypoints"])
        scores = keypoints[:, 2::3]
        scores = np.sum(scores, axis=1)
        keep = np.where(scores != 0)[0]

        roi_rec["keypoints"] = keypoints[keep, :].tolist()
        roi_rec["boxes"] = np.array(roi_rec["boxes"])[keep, :].tolist()

        if "gt_classes" in roi_rec:
            roi_rec["gt_classes"] = np.array(roi_rec["gt_classes"])[
                keep
            ].tolist()  # noqa

        if np.sum(scores) == 0:
            print("catch a empty box")
            return None
        else:
            return roi_rec


@OBJECT_REGISTRY.register
class KeyPointAnnoNumTs(object):
    def __init__(self, class_name, sub_class_name, ts_sub_class_name):
        self.class_name = class_name
        self.sub_class_name = sub_class_name
        self.ts_sub_class_name = ts_sub_class_name

    def _anno_trans(self, kps_points, kps_attr, matched_bbox_attrs):
        NotImplemented

    def _get_match_boxes(self, anno):
        data_bbox = anno[self.class_name]
        bbox_dict = {}
        for bbox_ele in data_bbox:
            if "id" not in bbox_ele.keys():
                continue
            bbox_dict[int(bbox_ele["id"])] = bbox_ele

        matching = anno["belong_to"]
        matching_dict = {}
        for matching_ele in matching:
            matching_list = matching_ele.strip().split(":")
            bbox_id = int(matching_list[0].strip().split("|")[-1])
            kps_id = int(matching_list[1].strip().split("|")[-1])
            matching_dict[kps_id] = bbox_id

        return bbox_dict, matching_dict

    def __call__(self, item):
        (x, anno) = item
        anno = anno.copy()
        if (
            self.class_name not in anno.keys()
            or self.sub_class_name not in anno.keys()
        ):
            return ((x, anno),)
        if "belong_to" not in anno.keys():
            return ((x, anno),)

        bbox_dict, matching_dict = self._get_match_boxes(anno)

        ts_kps_data = []
        og_kps_data = anno[self.sub_class_name]

        for kps_ele in og_kps_data:
            if "id" not in kps_ele.keys():
                continue
            if "point_attrs" not in kps_ele.keys():
                continue
            if "data" not in kps_ele.keys():
                continue

            kps_id = kps_ele["id"]
            kps_attr = kps_ele["point_attrs"]
            kps_points = kps_ele["data"]

            if kps_id not in matching_dict.keys():
                continue

            matched_bbox = bbox_dict[matching_dict[kps_id]]
            matched_bbox_attrs = matched_bbox["attrs"]
            ts_kps_points, ts_kps_attr = self._anno_trans(
                kps_points, kps_attr, matched_bbox_attrs
            )

            if len(ts_kps_points) == 0:
                continue

            kps_ele["point_attrs"] = ts_kps_attr
            kps_ele["data"] = ts_kps_points
            ts_kps_data.append(kps_ele)
        if len(ts_kps_data) != 0:
            anno[self.ts_sub_class_name] = ts_kps_data
            anno.pop(self.sub_class_name)
            anno["belong_to"] = [
                x.replace(self.sub_class_name, self.ts_sub_class_name)
                for x in anno["belong_to"]
            ]
        return ((x, anno),)


@OBJECT_REGISTRY.register
class VehicleKps8to2Ts(KeyPointAnnoNumTs):
    def __init__(self, class_name, sub_class_name, ts_sub_class_name):
        super(VehicleKps8to2Ts, self).__init__(
            class_name=class_name,
            sub_class_name=sub_class_name,
            ts_sub_class_name=ts_sub_class_name,
        )

    def _anno_trans(self, kps_points, kps_attr, matched_bbox_attrs):
        offset = 10
        kps_occ = list(
            map(lambda point: point["point_label"]["occlusion"], kps_attr)
        )
        kps_ignore = list(
            map(lambda point: point["point_label"]["ignore"], kps_attr)
        )
        truncation = matched_bbox_attrs.get("truncation", "None")

        if len(kps_points) < 4 or len(kps_ignore) < 4:
            # return [], []
            return kps_points, kps_attr

        if truncation != "None":
            # box with truncation, vehicle partly visible
            left_pt_valid = True
            left_angle_valid = True
            left_attr_valid = kps_ignore[0] != "yes" or kps_ignore[1] != "yes"
            right_pt_valid = True
            right_angle_valid = True
            right_attr_valid = kps_ignore[3] != "yes" or kps_ignore[2] != "yes"
            if left_attr_valid:
                if kps_ignore[0] != "yes":
                    occ_valid = kps_occ[0] != "self_occluded"
                elif kps_ignore[1] != "yes":
                    occ_valid = kps_occ[1] != "self_occluded"
                _left_occ_valid = occ_valid
            else:
                _left_occ_valid = False
            if right_attr_valid:
                if kps_ignore[3] != "yes":
                    occ_valid = kps_occ[3] != "self_occluded"
                elif kps_ignore[2] != "yes":
                    occ_valid = kps_occ[2] != "self_occluded"
                _right_occ_valid = occ_valid
            else:
                _right_occ_valid = False
            # if the whole left or right part is outside the image, skip
            left_occ_valid = _left_occ_valid and right_attr_valid
            right_occ_valid = left_attr_valid and _right_occ_valid
        else:
            # box without truncation, vehicle fully visible
            # point valid
            left_pt_valid = kps_points[1] != kps_points[0]
            right_pt_valid = kps_points[2] != kps_points[3]
            # attribute valid
            left_attr_valid = kps_ignore[0] != "yes" and kps_ignore[1] != "yes"
            right_attr_valid = (
                kps_ignore[3] != "yes" and kps_ignore[2] != "yes"
            )
            # angle valid
            vh_vector_left = (
                kps_points[1][0] - kps_points[0][0],
                kps_points[1][1] - kps_points[0][1],
            )
            angle = math.atan2(vh_vector_left[1], vh_vector_left[0])
            angle_left = angle * 180 / math.pi
            angle_left = (angle_left + 360) % 180
            left_angle_valid = angle_left > (90 + offset) or angle_left < (
                90 - offset
            )
            vh_vector_right = (
                kps_points[2][0] - kps_points[3][0],
                kps_points[2][1] - kps_points[3][1],
            )
            angle = math.atan2(vh_vector_right[1], vh_vector_right[0])
            angle_right = angle * 180 / math.pi
            angle_right = (angle_right + 360) % 180
            right_angle_valid = angle_right > (90 + offset) or angle_right < (
                90 - offset
            )

            # occ valid
            left_num_vis = int(kps_occ[1] != "self_occluded") + int(
                kps_occ[0] != "self_occluded"
            )
            right_num_vis = int(kps_occ[3] != "self_occluded") + int(
                kps_occ[2] != "self_occluded"
            )
            if left_num_vis > right_num_vis:
                left_occ_valid = True
                right_occ_valid = False
            elif left_num_vis < right_num_vis:
                left_occ_valid = False
                right_occ_valid = True
            else:
                if left_num_vis == 0:
                    left_occ_valid = False
                    right_occ_valid = False
                else:
                    # if the same num, decide by vehicle light points
                    left_num_vis = int(kps_occ[5] != "self_occluded") + int(
                        kps_occ[4] != "self_occluded"
                    )
                    right_num_vis = int(kps_occ[7] != "self_occluded") + int(
                        kps_occ[6] != "self_occluded"
                    )
                    if left_num_vis > right_num_vis:
                        left_occ_valid = True
                        right_occ_valid = False
                    elif left_num_vis < right_num_vis:
                        left_occ_valid = False
                        right_occ_valid = True
                    else:
                        left_occ_valid = False
                        right_occ_valid = False
        # overall decision
        left_valid = (
            left_pt_valid
            and left_attr_valid
            and left_angle_valid
            and left_occ_valid
        )
        right_valid = (
            right_pt_valid
            and right_attr_valid
            and right_angle_valid
            and right_occ_valid
        )

        if left_valid:
            return [kps_points[0], kps_points[1]], [kps_attr[0], kps_attr[1]]
        elif right_valid:
            return [kps_points[3], kps_points[2]], [kps_attr[3], kps_attr[2]]
        else:
            return [], []


@OBJECT_REGISTRY.register
class DenseBoxSubboxDetAnnoTs(object):
    """
    Default annotation transformer.

    for subbox detection that packed in the
    densebox image record format.

    Parameters
    ----------
    config : dict
        Configure
    root_dir : str
        Image root
    """

    def __init__(self, config, root_dir, verbose=True, skip_invalid=True):
        self.verbose = verbose
        self.root_dir = root_dir
        self.config = config
        self.skip_invalid = skip_invalid

    def _get_20_points(self, bbox_1, bbox_2):
        points_data = _get_box_10_points(bbox_1) + _get_box_10_points(
            bbox_2
        )  # noqa
        return points_data

    def __call__(self, item):
        if len(item) == 1 and item[0] is not None:
            image_dir, anno = item[0]
        elif len(item) == 2:
            image_dir, anno = item
        else:
            return None

        instances = []
        ignore_regions = []
        image_url = os.path.join(image_dir, anno["image_key"])

        if not os.path.exists(image_url):
            if self.skip_invalid:
                if self.verbose:
                    warnings.warn(
                        "WARNING: skip invalid image: %s" % (image_url)
                    )
                return None
            else:
                raise RuntimeError("No such image: %s" % (image_url))

        if not _check_image_completeness(image_url):
            if self.verbose:
                warnings.warn(
                    "WARNING: skip premature end image: %s" % (image_url)
                )  # noqa
            return None

        parent_boxes = []
        for obj in anno.get(self.config["parent_box_classname"], []):
            x1, y1, x2, y2 = map(float, obj["data"])
            height = y2 - y1
            width = x2 - x1
            if height <= 0 or width <= 0:
                continue
            if check_obj(obj, self.config.get("parent_box_remove_condiction")):
                continue
            parent_boxes.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "matched": False,
                    "id": obj.get("id", ""),
                }
            )

        children_boxes = []
        for obj in anno.get(self.config["children_box_classname"], []):
            x1, y1, x2, y2 = map(float, obj["data"])
            height = y2 - y1
            width = x2 - x1
            if height <= 0 or width <= 0:
                continue
            if check_obj(
                obj, self.config.get("children_box_remove_condiction")
            ):  # noqa
                continue
            if check_obj(
                obj, self.config.get("children_box_ignore_condiction")
            ):  # noqa
                ignore_region = {
                    "left_top": [x1, y1],
                    "right_bottom": [x2, y2],
                    "class_id": [self.config["current_class_id"]],
                }
                ignore_regions.append(ignore_region)
            elif check_obj(
                obj, self.config.get("children_box_hard_condiction")
            ):  # noqa
                children_boxes.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "hard": True,
                        "matched": False,
                        "id": obj.get("id", ""),
                    }
                )
            elif check_obj(
                obj, self.config.get("children_box_positive_condiction")
            ):
                children_boxes.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "hard": False,
                        "matched": False,
                        "id": obj.get("id", ""),
                    }
                )
            else:
                warnings.warn(
                    "WARNING, not matched obj: %s" % (json.dumps(obj))
                )  # noqa

        if self.config["match_mode"] == "matching_with_overlaps":
            if len(parent_boxes) and len(children_boxes):
                matched_results = matching_with_overlaps(
                    list(map(lambda x: x["bbox"], parent_boxes)),
                    list(map(lambda x: x["bbox"], children_boxes)),
                )
                for parent_id, children_id, overlap in matched_results:
                    if overlap > self.config["match_overlap_threshold"]:
                        parent_box = parent_boxes[parent_id]
                        parent_box["matched"] = True
                        children_box = children_boxes[children_id]
                        children_box["matched"] = True
                        points_data = self._get_20_points(
                            parent_box["bbox"], children_box["bbox"]
                        )
                        instance = {
                            "points_data": points_data,
                            "class_id": [self.config["current_class_id"]],
                            "attribute": [],
                            "is_hard": [int(children_box["hard"])],
                        }
                        instances.append(instance)
            for parent_box in parent_boxes:
                if not parent_box["matched"]:
                    points_data = self._get_20_points(
                        parent_box["bbox"], [-10000, -10000, -10000, -10000]
                    )
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [False],
                    }
                    instances.append(instance)
            for children_box in children_boxes:
                if not children_box["matched"]:
                    points_data = self._get_20_points(
                        children_box["bbox"], children_box["bbox"]
                    )
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [int(children_box["hard"])],
                    }
                    instances.append(instance)
        elif (
            self.config["match_mode"]
            == "pack_children_then_parent_and_ignore_unmatched_parent"
        ):  # noqa
            if len(parent_boxes) and len(children_boxes):
                matched_results = matching_with_overlaps(
                    list(map(lambda x: x["bbox"], parent_boxes)),
                    list(map(lambda x: x["bbox"], children_boxes)),
                )
                for parent_id, children_id, overlap in matched_results:
                    if overlap > self.config["match_overlap_threshold"]:
                        parent_box = parent_boxes[parent_id]
                        parent_box["matched"] = True
                        children_box = children_boxes[children_id]
                        children_box["matched"] = True
                        points_data = self._get_20_points(
                            children_box["bbox"], parent_box["bbox"]
                        )
                        instance = {
                            "points_data": points_data,
                            "class_id": [self.config["current_class_id"]],
                            "attribute": [],
                            "is_hard": [int(children_box["hard"])],
                        }
                        instances.append(instance)
                for parent_box in parent_boxes:
                    if not parent_box["matched"]:
                        x1, y1, x2, y2 = parent_box["bbox"]
                        ignore_region = {
                            "left_top": [x1, y1],
                            "right_bottom": [x2, y2],
                            "class_id": [self.config["parent_class_id"]],
                        }
                        ignore_regions.append(ignore_region)
        elif (
            self.config["match_mode"]
            == "pack_parent_then_children_and_ignore_unmatched_children"
        ):  # noqa
            if len(parent_boxes) and len(children_boxes):
                matched_results = matching_with_overlaps(
                    list(map(lambda x: x["bbox"], parent_boxes)),
                    list(map(lambda x: x["bbox"], children_boxes)),
                )
                for parent_id, children_id, overlap in matched_results:
                    if overlap > self.config["match_overlap_threshold"]:
                        parent_box = parent_boxes[parent_id]
                        parent_box["matched"] = True
                        children_box = children_boxes[children_id]
                        children_box["matched"] = True
                        points_data = self._get_20_points(
                            parent_box["bbox"], children_box["bbox"]
                        )
                        instance = {
                            "points_data": points_data,
                            "class_id": [self.config["current_class_id"]],
                            "attribute": [],
                            "is_hard": [int(children_box["hard"])],
                        }
                        instances.append(instance)
            for parent_box in parent_boxes:
                if not parent_box["matched"]:
                    points_data = self._get_20_points(
                        parent_box["bbox"], [-10000, -10000, -10000, -10000]
                    )
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [False],
                    }
                    instances.append(instance)
            for children_box in children_boxes:
                if not children_box["matched"]:
                    x1, y1, x2, y2 = map(float, children_box["bbox"])
                    ignore_region = {
                        "left_top": [x1, y1],
                        "right_bottom": [x2, y2],
                        "class_id": [self.config["current_class_id"]],
                    }
                    ignore_regions.append(ignore_region)
        elif self.config["match_mode"] == "matching_with_belongto_attr":
            belong_to_list = anno.get("belong_to", [])
            if len(belong_to_list) != 0:
                matched_results = matching_with_belongto_attr(
                    parent_boxes, children_boxes, belong_to_list, self.config
                )  # noqa
                for parent_id, children_id in matched_results:
                    parent_box = parent_boxes[parent_id]
                    parent_box["matched"] = True
                    children_box = children_boxes[children_id]
                    children_box["matched"] = True
                    points_data = self._get_20_points(
                        parent_box["bbox"], children_box["bbox"]
                    )  # noqa
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [int(children_box["hard"])],
                    }
                    instances.append(instance)
            for parent_box in parent_boxes:
                if not parent_box["matched"]:
                    points_data = self._get_20_points(
                        parent_box["bbox"], [-10000, -10000, -10000, -10000]
                    )  # noqa
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [False],
                    }
                    instances.append(instance)
        elif (
            self.config["match_mode"]
            == "matching_with_belongto_attr_and_remove_unmatched_parent"
        ):  # noqa
            belong_to_list = anno.get("belong_to", [])
            if len(belong_to_list) != 0:
                matched_results = matching_with_belongto_attr(
                    parent_boxes, children_boxes, belong_to_list, self.config
                )  # noqa
                for parent_id, children_id in matched_results:
                    parent_box = parent_boxes[parent_id]
                    parent_box["matched"] = True
                    children_box = children_boxes[children_id]
                    children_box["matched"] = True
                    points_data = self._get_20_points(
                        parent_box["bbox"], children_box["bbox"]
                    )  # noqa
                    instance = {
                        "points_data": points_data,
                        "class_id": [self.config["current_class_id"]],
                        "attribute": [],
                        "is_hard": [int(children_box["hard"])],
                    }
                    instances.append(instance)
        else:
            raise Exception(
                "Invalid match mode: %s" % self.config["match_mode"]
            )

        if self.config.get("remove_empty_images", False) and not len(
            instances
        ):  # noqa
            return None
        np.random.shuffle(instances)
        img = cv2.imread(image_url, cv2.IMREAD_UNCHANGED)
        if img is None:
            if self.verbose:
                warnings.warn("WARNING: skip invalid image: %s" % (image_url))
            return None
        img_url = os.path.relpath(image_url, self.root_dir)
        if self.config.get("remove_zh_image_path", False) and re.findall(
            "[\u4e00-\u9fa5]", img_url
        ):  # noqa
            if self.verbose:
                warnings.warn(
                    "WARNING, deprecated params: skip zh image path: %s"
                    % (image_url)
                )  # noqa
            return None
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_c = img.shape[2] if len(img.shape) == 3 else 1

        if self.config.get("default_ignore_full_image", False):
            for class_id in range(1, self.config["num_classes"] + 1):
                if class_id != self.config["current_class_id"]:
                    ignore_region = {
                        "left_top": (0, 0),
                        "right_bottom": (img_w, img_h),
                        "class_id": [class_id],
                    }
                    ignore_regions.append(ignore_region)

        img_dict = {
            "img_url": img_url,
            "img_h": img_h,
            "img_w": img_w,
            "img_c": img_c,
            "instances": instances,
            "ignore_regions": ignore_regions,
        }

        return img_dict


def matching_with_overlaps(parent_bboxes, children_bboxes):
    """
    Bipartite graph matching.

    with negative IoU between parent bounding bboxes
    and children bounding boxes

    Parameters
    ----------
    parent_bboxes : list of list
        List of parent bounding bbox
    children_bboxes : list of list
        List of children bounding bbox
    """

    from sklearn.utils.linear_assignment_ import linear_assignment

    parent_bbox_areas = cal_bbox_areas(parent_bboxes)
    children_bbox_areas = cal_bbox_areas(children_bboxes)
    intersection_bbox_areas_matrix = cal_intersection_areas(
        parent_bboxes, children_bboxes
    )
    union_areas_matrix = (
        parent_bbox_areas.reshape(-1, 1)
        + children_bbox_areas.reshape(1, -1)
        - intersection_bbox_areas_matrix
    )
    contains_matrix = (
        intersection_bbox_areas_matrix / children_bbox_areas.reshape(1, -1)
    )
    ious_matrix = intersection_bbox_areas_matrix / union_areas_matrix
    matched_pairs = linear_assignment(-ious_matrix)
    results = []
    for parent_id, children_id in matched_pairs:
        overlap = contains_matrix[parent_id, children_id]
        results.append((parent_id, children_id, overlap))
    return results


def cal_bbox_areas(bboxes):
    """
    Calculate areas of a group of bounding boxes.

    Parameters
    ----------
    bboxes : list of list
        List of bounding bbox
    """
    bboxes = np.asarray(bboxes)
    ws = np.maximum(bboxes[:, 2] - bboxes[:, 0] + 1, 0)
    hs = np.maximum(bboxes[:, 3] - bboxes[:, 1] + 1, 0)
    return ws * hs


def image_fail_parsing_anno_to_contours_fn(
    ct_type, ct, width, height, labels, imgname
):
    """
    Transfrom the annotation data to contours for image fail parsing.

    Parameters
    ----------
    ct_type: int
        the contour type of image fail parsing, see Enum ImageFailCtsType.
    ct: dict
        the contour data.
    partial_certain_cts: list
        all partial certain contours.
    ignore_cts: list
        all ignore contours.
    width: int
        the image width.
    height: int
        the image height.
    labels: dict
        the label map.
    """
    pts = ct["data"]
    if None in pts:
        return None
    ct_pts = []
    for pt in pts:
        ct_pts.append(
            (
                max(min(int(float(pt[0])), width - 1), 0),
                max(min(int(float(pt[1])), height - 1), 0),
            )
        )
    ct_pts = np.array(ct_pts)

    value = 0
    anno_type = ct["attrs"]["type"]
    anno_degree = ct["attrs"]["degree"]
    if anno_type == "normal":
        anno_label_name = anno_type
    else:
        anno_label_name = f"{anno_degree}_{anno_type}"
    if ct_type == ImageFailCtsType.FULL_IMAGE:
        if anno_label_name in labels:
            value = labels[anno_label_name]
        else:
            value = 255
        ct_pts = [(0, 0), (0, height), (width, height), (width, 0)]
        ct_pts = np.array(ct_pts)
    elif ct_type == ImageFailCtsType.PARTIAL_POSSIBLE:
        # ignore for all transition area between possible and certain
        value = 255
    elif ct_type == ImageFailCtsType.PARTIAL_CERTAIN:
        if anno_label_name in labels:
            value = labels[anno_label_name]
        else:
            value = 255
    elif ct_type == ImageFailCtsType.IGNORE:
        value = 255
        full_image = ct["attrs"]["photo"]
        if full_image == "yes":
            ct_pts = [(0, 0), (0, height), (width, height), (width, 0)]
            ct_pts = np.array(ct_pts)
    # check night_light
    # anno_night_light = 'no_light'
    # if 'night_light' in ct['attrs']:
    #     anno_night_light = ct['attrs']['night_light']
    #     if anno_type == 'blur' and anno_night_light == 'dark_night_light':
    #         value = labels['normal']
    assert value != -1, "{}, {}".format(
        ct["attrs"]["type"], ct["attrs"]["ignore"]
    )  # noqa
    return ct_pts, value


@unique
class ImageFailCtsType(Enum):
    FULL_IMAGE = 1
    PARTIAL_POSSIBLE = 2
    PARTIAL_CERTAIN = 3
    IGNORE = 4


def anno_to_contours_fn_with_label_mapping(
    data, width, height, label_map_dict
):
    """
    Transfrom the annotation data to contours.

    Parameters
    ----------
    data: dict
        the annotation label.
    width: int
        the image width.
    height: int
        the image height.
    label_map_dict:
        the label map.
    """
    pts = data["data"]
    if None in pts:
        return None
    ct_pts = []
    for pt in pts:
        ct_pts.append(
            (
                max(min(int(float(pt[0])), width - 1), 0),
                max(min(int(float(pt[1])), height - 1), 0),
            )
        )
    ct_pts = np.array(ct_pts)
    match_flag, value = is_label_match(data["attrs"], label_map_dict)
    assert match_flag, "data attrs dont match with label mapping conditions"
    return ct_pts, value


def is_label_match(data_attrs, label_map_dict):
    """
    Get the mapping according to conditions of label map dict.

    Parameters
    ----------
    data_attrs: dict
        the labeled information.
    label_map_dict: dict
        the label mao dict including match conditions and label value.
    """
    for label_name in label_map_dict.keys():
        match_conditions_list = label_map_dict[label_name]["match_conditions"]
        value = label_map_dict[label_name]["id"]
        for match_conditions in match_conditions_list:
            for match_key in match_conditions.keys():
                match_flag = False
                if data_attrs[match_key] in match_conditions[match_key]:
                    match_flag = True
                else:
                    match_flag = False
                    break
            if match_flag is True:
                return match_flag, value
    if match_flag is False:
        return match_flag, None


def cal_intersection_areas(lhs_bboxes, rhs_bboxes):
    """
    Calculate areas of intersection boxes between two group of bounding boxes.

    Parameters
    ----------
    lhs_bboxes : list of list
        List of bounding bbox
    rhs_bboxes : list of list
        List of bounding bbox
    """
    if not len(lhs_bboxes) or not len(rhs_bboxes):
        iou_matrix = np.array([]).reshape(len(lhs_bboxes), len(rhs_bboxes))
        return iou_matrix

    lhs_bboxes = np.asarray(lhs_bboxes)
    rhs_bboxes = np.asarray(rhs_bboxes)

    lhs_x1 = lhs_bboxes[:, 0].reshape(-1, 1)
    rhs_x1 = rhs_bboxes[:, 0].reshape(1, -1)

    lhs_y1 = lhs_bboxes[:, 1].reshape(-1, 1)
    rhs_y1 = rhs_bboxes[:, 1].reshape(1, -1)

    lhs_x2 = lhs_bboxes[:, 2].reshape(-1, 1)
    rhs_x2 = rhs_bboxes[:, 2].reshape(1, -1)

    lhs_y2 = lhs_bboxes[:, 3].reshape(-1, 1)
    rhs_y2 = rhs_bboxes[:, 3].reshape(1, -1)

    i_x1 = np.maximum(lhs_x1, rhs_x1)
    i_y1 = np.maximum(lhs_y1, rhs_y1)
    i_x2 = np.minimum(lhs_x2, rhs_x2)
    i_y2 = np.minimum(lhs_y2, rhs_y2)

    i_ws = np.maximum(i_x2 - i_x1 + 1.0, 0.0)
    i_hs = np.maximum(i_y2 - i_y1 + 1.0, 0.0)
    i_areas = i_ws * i_hs
    return i_areas


def matching_with_belongto_attr(
    parent_bboxes, children_bboxes, belong_to_list, config
):  # noqa
    """
    Matching with "belongto" attribute between pareng bounding boxes and
    children bounding boxes

    Parameters
    ----------
    parent_bboxes : list of list
        List of parent bounding bbox
    children_bboxes : list of list
        List of children bounding bbox
    """
    results = []
    parent_mapping_dict = {}
    children_mapping_dict = {}

    for parent_idx, parent_bbox in enumerate(parent_bboxes):
        parent_mapping_dict[parent_bbox["id"]] = parent_idx
    for children_idx, children_bbox in enumerate(children_bboxes):
        children_mapping_dict[children_bbox["id"]] = children_idx

    for belong_to in belong_to_list:
        parent_str, children_str = belong_to.split(":")
        parent_cls_name, parent_box_id = parent_str.split("|")
        children_cls_name, children_box_id = children_str.split("|")

        if (
            parent_cls_name != config["parent_box_classname"]
            or children_cls_name != config["children_box_classname"]
        ):
            continue

        parent_id = parent_mapping_dict.get(int(parent_box_id), None)
        children_id = children_mapping_dict.get(int(children_box_id), None)

        if parent_id is not None and children_id is not None:
            results.append((parent_id, children_id))

    return results


@OBJECT_REGISTRY.register
class VehicleFlankAnnoTs(object):
    """
    Annotation transformer for Vehicle-Flank detection.

    which format the
    ground-line and side-edge to flank-data

    Parameters
    ----------
    parent_classname : str
        the classname of vehicle
    child_classname : str
        the classname of vehicle flank
    anno_adapter : AnnoAdapter
        the adapter for different annotated dataset,
        such as key-points 8 dataset
    min_flank_width : float
        the min valid flank width for instance to be packed
    empty_value : float
        the default value used to fill empty flank
    vis_dir : str
        if is not None, save the transform visualize result to this dir
    """

    def __init__(
        self,
        parent_classname="vehicle",
        child_classname="vehicle_flank",
        anno_adapter=None,
        min_flank_width=4,
        empty_value=-10000,
        vis_dir=None,
    ):
        self.parent_classname = parent_classname
        self.child_classname = child_classname
        if anno_adapter is None:
            self.anno_adapter = None
        elif isinstance(anno_adapter, dict):
            anno_adapter = build_from_registry(anno_adapter)  # noqa
            assert isinstance(anno_adapter, AnnoAdapter)
            self.anno_adapter = anno_adapter
        elif isinstance(anno_adapter, AnnoAdapter):
            self.anno_adapter = anno_adapter
        else:
            raise NotImplementedError
        self.min_flank_width = min_flank_width
        self.empty_value = empty_value
        self.vis_dir = vis_dir

    def __call__(self, item):
        """
        Parameters.

        ----------
        item : tuple
            which contains <image, anno>

        Returns
        -------
        record : dict
            The record for one image to be packed
            {
                'image': image_url
                'height': image_height
                'width': image_width
                'bboxes': each:
                    {
                        'data': <x1,y1,x2,y2>
                        'class_id':
                    }
                'flanks': each:
                    {
                        'data': <left_bottom_point, right_bottom_point,
                                 right_top_point, left_top_point>
                        'class_id':
                    }
                'ignore_regions':
                    {
                        'left_top': bbox[0:2],
                        'right_bottom': bbox[2:4],
                        'class_id':
                    }
            }
        """
        if self.anno_adapter is not None:
            item = self.anno_adapter(item)
        img_dir, anno = item

        img_url = os.path.join(img_dir, anno["image_key"])
        image_url = os.path.abspath(img_url)
        if image_url is None:
            return None

        bbox_lst, flank_lst, ignore_lst = [], [], []
        parent_map = get_instance_map(anno, self.parent_classname)
        child_map = get_instance_map(anno, self.child_classname)
        couples = get_couples(
            anno, self.parent_classname, self.child_classname
        )  # noqa
        parent_id_coupled = set()
        # process the coupled vehicle and flank
        for parent_id, child_id in couples:
            parent = parent_map.get(parent_id, None)
            child = child_map.get(child_id, None)
            if parent is None or child is None:
                continue

            parent_id_coupled.add(parent_id)
            x1, y1, x2, y2 = [float(val) for val in parent["data"]]
            if (
                parent["attrs"]["ignore"].lower() == "yes"
                or child["attrs"]["faced_flank"] == "unknown"
            ):
                # ignore
                ignore_lst.append(
                    {
                        "left_top": [x1, y1],
                        "right_bottom": [x2, y2],
                        "class_id": [float(1)],
                    }
                )
            elif child["attrs"]["faced_flank"] in ["rear", "head"]:
                # negative
                bbox_lst.append(
                    {"data": [x1, y1, x2, y2], "class_id": float(1)}
                )
                flank_lst.append({"data": child["data"], "class_id": float(0)})
            elif child["attrs"]["faced_flank"] in ["left", "right"]:
                flank_values = []
                for point in child["data"]:
                    flank_values += point
                if self.empty_value in flank_values:
                    # ignore
                    ignore_lst.append(
                        {
                            "left_top": [x1, y1],
                            "right_bottom": [x2, y2],
                            "class_id": [float(1)],
                        }
                    )
                else:
                    left_bottom, right_bottom = child["data"][0:2]
                    if right_bottom[0] - left_bottom[0] < self.min_flank_width:
                        # ignore
                        ignore_lst.append(
                            {
                                "left_top": [x1, y1],
                                "right_bottom": [x2, y2],
                                "class_id": [float(1)],
                            }
                        )
                    else:
                        # positive
                        bbox_lst.append(
                            {"data": [x1, y1, x2, y2], "class_id": float(1)}
                        )
                        flank_lst.append(
                            {"data": child["data"], "class_id": float(1)}
                        )
            else:
                raise NotImplementedError

        # no valid flank instance, just delete the images
        if len(flank_lst) == 0:
            return None

        # process the uncoupled vehicle to ignore
        for parent_id, parent in parent_map.items():
            if parent_id in parent_id_coupled:
                continue
            x1, y1, x2, y2 = [float(val) for val in parent["data"]]
            ignore_lst.append(
                {
                    "left_top": [x1, y1],
                    "right_bottom": [x2, y2],
                    "class_id": [float(1)],
                }
            )

        # assign label
        record = {
            "image": image_url,
            "height": anno["height"],
            "width": anno["width"],
            "bboxes": bbox_lst,
            "flanks": flank_lst,
            "ignore_regions": ignore_lst,
        }

        return record


@OBJECT_REGISTRY.register
class AnnoAdapter(object):
    def __init__(self, parent_classname, child_classname, target_classname):
        """
        Anno-adapter interface.

        Parameters
        ----------
        parent_classname : str
            the parent bbox classname; such as 'vehicle'
        child_classname : str
            the child instance classname of the origin anno dataset,
            such as 'vehicle_kps_8'
        target_classname : str
            the target instance class name, such as 'vehicle_flank'
        """
        self.parent_classname = parent_classname
        self.child_classname = child_classname
        self.target_classname = target_classname

    def __call__(self, item):
        x, anno = item
        new_anno = copy.deepcopy(anno)

        parent_map = get_instance_map(anno, self.parent_classname)
        couples = get_couples(
            anno, self.child_classname, self.parent_classname
        )  # noqa
        couples_map = {one_id: other_id for one_id, other_id in couples}

        new_child_lst = []
        new_belong_to_lst = []
        for child in anno.get(self.child_classname, []):
            child_id = child["id"]
            parent_id = couples_map[child_id]
            parent = parent_map.get(parent_id, None)
            if parent is None or child is None:
                continue
            try:
                child = self.adapt(parent, child)
            except (KeyError, IndexError):
                # some error data exist on anno dataset
                continue
            new_child_lst.append(child)
            belong_to = f"{self.parent_classname}|{parent_id}:{self.target_classname}|{child_id}"  # noqa
            new_belong_to_lst.append(belong_to)
        new_anno[self.target_classname] = new_child_lst
        new_anno["belong_to"] = new_belong_to_lst

        return x, new_anno

    def adapt(self, parent, child) -> dict:
        """
        Process coupled parent and child to flank instance.

        Parameters
        ----------
        parent : dict
            the instance dict of vehicle bbox
        child : dict
            the instance dict of child(such as key-points 8)

        Returns
        -------
        vehicle_flank: dict
            {
                'id':
                'data': [left_bottom_point, right_bottom_point,
                         left_top_point, right_top_point]
                'attrs': {
                    'faced_flank': value in ['right','left','head','rear','unknown']    # noqa
                }
            }

        """
        raise NotImplementedError


@OBJECT_REGISTRY.register
class AnnoAdapterKps8(AnnoAdapter):
    def __init__(
        self,
        parent_classname,
        child_classname,
        target_classname,
        empty_value=-10000,
    ):
        super(AnnoAdapterKps8, self).__init__(
            parent_classname, child_classname, target_classname
        )  # noqa
        self.empty_value = empty_value

    def adapt(self, parent, child) -> dict:
        faced_flank = self.get_faced_flank_info(parent, child)
        vehicle_flank = self.get_vehicle_flank(parent, child, faced_flank)

        return vehicle_flank

    def get_vehicle_flank(self, parent, child, faced_flank) -> dict:
        """
        Parameters.

        ----------
        parent : dict
        child : dict
        faced_flank : str
            value in ['right','left','head','rear','unknown']

        Returns
        -------
        vehicle_flank : dict
            {
                'id':
                'data': [left_bottom_point, right_bottom_point,
                         left_top_point, right_top_point]
                'attrs': {
                    'faced_flank': value in ['right','left','head','rear','unknown']    # noqa
                }
            }

        """
        if faced_flank in ["head", "rear", "unknown"]:
            # NOTE: 1. 'head' and 'rear' are negative instance for flank task
            #       2. 'unknown' are ignore instance for flank task
            return {
                "id": child["id"],
                "data": [
                    [self.empty_value, self.empty_value] for _ in range(4)
                ],  # noqa
                "attrs": {"faced_flank": faced_flank},
            }
        elif faced_flank in ["left", "right"]:
            # NOTE: 'left' and 'right' are negative instance for flank task
            # while truncate the wheel out, you need ignore the instance for ground line task   # noqa
            (
                left_bottom_point,
                right_bottom_point,
                left_top_point,
                right_top_point,
            ) = self.get_vehicle_flank_data(parent, child, faced_flank)
            return {
                "id": child["id"],
                "data": [
                    left_bottom_point,
                    right_bottom_point,
                    left_top_point,
                    right_top_point,
                ],
                "attrs": {"faced_flank": faced_flank},
            }
        else:
            raise ValueError("Error faced_flank: {}".format(faced_flank))

    def get_vehicle_flank_data(self, parent, child, faced_flank):
        """
        Parameters.

        ----------
        parent : dict
        child : dict
        faced_flank : str


        """
        assert faced_flank in ["right", "left"]
        x1, y1, x2, y2 = [float(val) for val in parent["data"]]
        points = child["data"]
        points_attr = child["point_attrs"]
        if faced_flank == "left":
            left_wheel_idx, right_wheel_idx = 1, 0
            left_edge_idx, right_edge_idx = 5, 4
        else:
            left_wheel_idx, right_wheel_idx = 3, 2
            left_edge_idx, right_edge_idx = 7, 6

        left_edge_ignore = self.is_ignore_point(points_attr[left_edge_idx])
        if left_edge_ignore:
            left_edge_x = x1
        else:
            left_edge_x = float(points[left_edge_idx][0])
        right_edge_ignore = self.is_ignore_point(points_attr[right_edge_idx])
        if right_edge_ignore:
            right_edge_x = x2
        else:
            right_edge_x = float(points[right_edge_idx][0])
        left_top_point = [float(left_edge_x), float(y1)]
        right_top_point = [float(right_edge_x), float(y1)]

        left_wheel_point = points[left_wheel_idx]
        left_wheel_ignore = self.is_ignore_point(points_attr[left_wheel_idx])
        right_wheel_point = points[right_wheel_idx]
        right_wheel_ignore = self.is_ignore_point(points_attr[right_wheel_idx])
        if left_wheel_ignore or right_wheel_ignore:
            # can`t get the vehicle ground line
            left_bottom_y = self.empty_value
            right_bottom_y = self.empty_value
        else:
            left_bottom_y, right_bottom_y = self.get_intersections_to_vertical(
                left_wheel_point,
                right_wheel_point,
                [left_edge_x, right_edge_x],
            )
        left_bottom_point = [float(left_edge_x), float(left_bottom_y)]
        right_bottom_point = [float(right_edge_x), float(right_bottom_y)]

        return (
            left_bottom_point,
            right_bottom_point,
            left_top_point,
            right_top_point,
        )  # noqa

    def get_intersections_to_vertical(self, one_point, other_point, loc_x_lst):
        one_x, one_y = one_point
        other_x, other_y = other_point
        delta_x = one_x - other_x
        delta_y = one_y - other_y
        if delta_x == 0:
            return [self.empty_value] * len(loc_x_lst)
        else:
            slope = delta_y / delta_x
            return [slope * (x - other_x) + other_y for x in loc_x_lst]

    def get_faced_flank_info(self, parent, child) -> str:
        """
        Parameters.

        ----------
        parent : dict
        child : dict

        Returns
        -------
        faced_flank : str
            value in ['right','left','head','rear','unknown']
        """
        child_attrs = child.get("attrs", {})
        if "faced_flank" in child_attrs:
            return child_attrs["faced_flank"]

        points = child["data"]
        points_attrs = child["point_attrs"]
        faced_flank_set = set()
        for start_idx, end_idx in [(0, 1), (3, 2), (4, 5), (7, 6)]:
            start_point, end_point = points[start_idx], points[end_idx]
            start_attr, end_attr = (
                points_attrs[start_idx],
                points_attrs[end_idx],
            )  # noqa
            faced = self.faced_flank_info(
                start_point, end_point, start_attr, end_attr
            )  # noqa
            if faced == "unknown":
                continue
            faced_flank_set.add(faced)
        if len(faced_flank_set) == 0:
            faced_flank = "unknown"
        elif len(faced_flank_set) == 1:
            faced_flank = faced_flank_set.pop()
        else:
            # TODO: diff the rear and head @feng02.li
            faced_flank = "rear"

        return faced_flank

    @staticmethod
    def faced_flank_info(start_point, end_point, start_attr, end_attr) -> str:
        """
        Parameters.

        ----------
        start_point
        end_point
        start_attr
        end_attr

        Returns
        -------
        faced_flank : str
            value in ['right','left','head','rear','unknown']

        """
        start_ignore = AnnoAdapterKps8.is_ignore_point(start_attr)
        end_ignore = AnnoAdapterKps8.is_ignore_point(end_attr)
        start_x, start_y = start_point
        end_x, end_y = end_point
        if not start_ignore and not end_ignore:
            if start_x < end_x:
                return "right"
            elif start_x > end_x:
                return "left"
            elif start_y > end_y:
                return "rear"
            else:
                return "head"
        else:
            return "unknown"

    @staticmethod
    def is_ignore_point(point_attr):
        """
        Parameters.

        ----------
        point_attr : dict
            {
                'occlusion':
                'Corner_confidence':
                'ignore':
                'position':
            }


        """
        ignore = point_attr.get("point_label", {}).get("ignore", "no").lower()

        return ignore == "yes"


def get_instance_map(anno, classname, valid_types=None):
    """
    Parameters.

    ----------
    anno : dict
    classname : dict
    valid_types : list

    Returns
    -------
    mapping : dict
        each record <instance_id, instance>
    """
    mapping = {}
    for instance in anno.get(classname, []):
        idx = instance.get("id", None)
        if idx is None or idx in mapping:
            continue
        if (
            valid_types is not None
            and instance["attrs"]["type"] not in valid_types
        ):
            continue
        mapping[int(idx)] = copy.deepcopy(instance)

    return mapping


def get_couples(anno, one_classname, other_classname):
    pairs = []
    for belong_to in anno.get("belong_to", []):
        one, other = belong_to.strip().split(":")
        one_name, one_id = one.split("|")
        other_name, other_id = other.split("|")
        tmp_map = {one_name: int(one_id), other_name: int(other_id)}
        parent_id = tmp_map.get(one_classname)
        child_id = tmp_map.get(other_classname)
        pairs.append((parent_id, child_id))

    return pairs
