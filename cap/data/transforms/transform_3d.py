import cv2
import numpy as np
from pycocotools.mask import decode, encode

from cap.registry import OBJECT_REGISTRY
from .detection import DetInputPadding
from .label_generator import (
    dense3d_pad_after_label_generator,
    label_encoding,
    roi_heatmap_label_encoding,
    roi_heatmap_label_encoding_undistort_uv_depth,
)
from .utils_3d import image_transform


@OBJECT_REGISTRY.register
class Image3DTransform(object):
    def __init__(
        self,
        input_wh,
        keep_res,
        shift=None,
        keep_aspect_ratio=False,
        support_wh=(2048, 1280),
    ):
        self.input_wh = input_wh
        self.keep_res = keep_res
        if shift is None:
            shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self._keep_aspect_ratio = keep_aspect_ratio
        self.support_wh = support_wh

    def __call__(self, data):
        img = data["img"]
        data["ori_img"] = img  # for debug
        orgin_wh = img.shape[:2][::-1]
        if self._keep_aspect_ratio and orgin_wh != self.support_wh:
            resize_wh_ratio = float(self.input_wh[0]) / float(
                self.input_wh[1]
            )  # noqa
            orgin_wh_ratio = float(orgin_wh[0]) / float(orgin_wh[1])
            affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])
            if resize_wh_ratio > orgin_wh_ratio:
                new_wh = (
                    int(orgin_wh[1] * resize_wh_ratio),
                    orgin_wh[1],
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
            elif resize_wh_ratio < orgin_wh_ratio:
                new_wh = (
                    orgin_wh[0],
                    int(orgin_wh[0] / resize_wh_ratio),
                )  # noqa
                img = cv2.warpAffine(img, affine, new_wh, 0)
        img, trans_matrix = image_transform(
            img, self.input_wh, self.keep_res, shift=self.shift
        )
        meta = data["anno"]["meta"] if "meta" in data["anno"] else {}
        meta["center"] = trans_matrix["center"]
        meta["size"] = trans_matrix["size"]
        meta["img_wh"] = np.array(img.shape[:2][::-1])
        meta["orgin_wh"] = np.array(orgin_wh)
        meta["trans_matrix"] = trans_matrix["trans_input"]
        data["img"] = img
        data["anno"]["meta"] = meta
        return data


@OBJECT_REGISTRY.register
class ImageTransformWithScale(object):
    def __init__(self, scale_wh, keep_res, shift=None):
        if shift is None:
            self.shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self.scale_wh = scale_wh
        self.keep_res = keep_res

    def __call__(self, data):
        img = data["img"]
        data["ori_img"] = img  # for debug
        orgin_wh = img.shape[:2][::-1]
        input_wh = (
            int(orgin_wh[0] * self.scale_wh[0]),
            int(orgin_wh[1] * self.scale_wh[1]),
        )

        img, trans_matrix = image_transform(
            img, input_wh, self.keep_res, shift=self.shift
        )
        meta = data["meta"] if "meta" in data else {}
        meta["center"] = trans_matrix["center"]
        meta["size"] = trans_matrix["size"]
        meta["img_wh"] = np.array(img.shape[:2][::-1])
        meta["orgin_wh"] = np.array(orgin_wh)
        meta["trans_matrix"] = trans_matrix["trans_input"]
        data["img"] = img
        data["meta"] = meta
        return data


@OBJECT_REGISTRY.register
class Heatmap3DDetectionLableGenerate(object):
    def __init__(
        self,
        num_classes,
        classid_map,
        normalize_depth,
        focal_length_default,
        alpha_in_degree,
        filtered_name,
        min_box_edge,
        max_depth,
        max_objs,
        down_stride=4,
        use_bbox2d=False,
        enable_ignore_area=False,
        use_project_bbox2d=False,
        shift=None,
        undistort_2dcenter=False,
        undistort_depth_uv=False,
        input_padding=None,
        keep_meta_keys=None,
    ):
        self.num_classes = num_classes
        self.classid_map = classid_map
        self.normalize_depth = normalize_depth
        self.focal_length_default = focal_length_default
        self.alpha_in_degree = alpha_in_degree
        self.filtered_name = filtered_name
        self.down_stride = down_stride
        self.use_bbox2d = use_bbox2d
        self.enable_ignore_area = enable_ignore_area
        if shift is None:
            self.shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self.min_box_edge = min_box_edge
        self.max_depth = max_depth
        self.max_objs = max_objs
        self.use_project_bbox2d = use_project_bbox2d
        self.undistort_2dcenter = undistort_2dcenter
        self.undistort_depth_uv = undistort_depth_uv
        self.input_padding = input_padding
        if keep_meta_keys is None:
            self.keep_meta_keys = ["img_wh"]
        else:
            self.keep_meta_keys = keep_meta_keys

    def __call__(self, data):

        label = data["anno"].pop("objects")
        meta = data["anno"]["meta"]
        meta['ori_img'] = data['ori_img']
        meta["calib"] = np.array(meta["calib"])
        gt = label_encoding(
            label,
            meta,
            self.num_classes,
            self.classid_map,
            self.normalize_depth,
            self.focal_length_default,
            self.alpha_in_degree,
            self.down_stride,
            self.use_bbox2d,
            self.enable_ignore_area,
            shift=self.shift,
            filtered_name=self.filtered_name,
            min_box_edge=self.min_box_edge,
            max_depth=self.max_depth,  # noqa
            max_objs=self.max_objs,
            use_project_bbox2d=self.use_project_bbox2d,  # noqa
            undistort_2dcenter=self.undistort_2dcenter,
            undistort_depth_uv=self.undistort_depth_uv,
        )

        # # add heatmap vis      zmj
        # heatmap = gt['heatmap']     # [c,h,w]
        # for cls_idx in range(self.num_classes):
        #     hm = heatmap[cls_idx,:,:]
        #     # [h,w,c]
        #     hm = (hm[:,:,np.newaxis] * 255).astype(np.int16)
        #     file_name = meta["file_name"]
        #     save_name = file_name + '_cls_' + str(cls_idx) + '.jpg'
        #     cv2.imwrite(save_name, hm)


        for k in list(meta.keys()):
            if k not in self.keep_meta_keys:
                meta.pop(k)
        if self.input_padding is not None:
            gt = dense3d_pad_after_label_generator(
                gt, self.input_padding, self.down_stride
            )  # noqa
        data = {"img": data["img"], **meta, **gt}
        data["img"] = data["img"].transpose(2, 0, 1)   # [h,w,c]  to [c,h,w]
        return data


@OBJECT_REGISTRY.register
class ROIHeatmap3DDetectionLableGenerate:
    def __init__(
        self,
        num_classes,
        classid_map,
        normalize_depth,
        focal_length_default,
        filtered_name,
        min_box_edge,
        max_depth,
        max_gt_boxes_num,
        is_train=True,
        use_bbox2d=False,
        use_project_bbox2d=False,
        undistort_depth_uv=False,
        shift=None,
        input_padding=None,
        keep_meta_keys=None,
    ):
        self.num_classes = num_classes
        self.classid_map = classid_map
        self.normalize_depth = normalize_depth
        self.focal_length_default = focal_length_default
        self.max_gt_boxes_num = max_gt_boxes_num
        self.use_bbox2d = use_bbox2d
        self.min_box_edge = min_box_edge
        self.use_project_bbox2d = use_project_bbox2d
        self.is_train = is_train
        if shift is None:
            self.shift = np.array([0, 0], dtype=np.float32)
        else:
            self.shift = shift
        self.max_depth = max_depth
        self.filtered_name = filtered_name
        self.undistort_depth_uv = undistort_depth_uv
        self.input_padding = input_padding
        if keep_meta_keys is None:
            self.keep_meta_keys = ["img_wh"]
        else:
            self.keep_meta_keys = keep_meta_keys

    def __call__(self, data):
        label = data["anno"].pop("objects")
        meta = data["anno"]["meta"]
        meta["calib"] = np.array(meta["calib"])
        if self.undistort_depth_uv:
            self.keep_meta_keys += ["eq_fu", "eq_fv"]
            gt = roi_heatmap_label_encoding_undistort_uv_depth(
                data["img"],
                label=label,
                meta=meta,
                num_classes=self.num_classes,
                classid_map=self.classid_map,
                normalize_depth=self.normalize_depth,
                focal_length_default=self.focal_length_default,
                max_gt_boxes_num=self.max_gt_boxes_num,
                filtered_name=self.filtered_name,
                use_bbox2d=self.use_bbox2d,
                shift=self.shift,
                max_depth=self.max_depth,
                use_project_bbox2d=self.use_project_bbox2d,
            )
            meta["im_hw"] = meta["img_wh"][::-1].astype(np.float32)
            distCoeffs = meta["distCoeffs"]
            for k in list(meta.keys()):
                if k not in self.keep_meta_keys:
                    meta.pop(k)
        else:
            gt = roi_heatmap_label_encoding(
                data["img"],
                label=label,
                meta=meta,
                num_classes=self.num_classes,
                classid_map=self.classid_map,
                normalize_depth=self.normalize_depth,
                focal_length_default=self.focal_length_default,
                max_gt_boxes_num=self.max_gt_boxes_num,
                filtered_name=self.filtered_name,
                use_bbox2d=self.use_bbox2d,
                shift=self.shift,
                max_depth=self.max_depth,
                use_project_bbox2d=self.use_project_bbox2d,
            )
            meta["im_hw"] = meta["img_wh"][::-1].astype(np.float32)

            distCoeffs = meta["distCoeffs"]
            for k in list(meta.keys()):
                if k not in self.keep_meta_keys:
                    meta.pop(k)
        if self.is_train:
            data = {
                "img": data["img"],
                **meta,
                **gt,
                "distCoeffs": np.array(distCoeffs),
            }
        else:
            ignore_mask = decode(meta["ignore_mask"]).astype(np.uint8)
            trans_matrix = meta.pop("trans_matrix")
            new_wh = (
                int(ignore_mask.shape[1] * trans_matrix[0][0]),
                int(ignore_mask.shape[0] * trans_matrix[1][1]),
            )
            ignore_mask = cv2.warpAffine(ignore_mask, trans_matrix, new_wh, 0)
            meta["ignore_mask"] = encode(np.asfortranarray(ignore_mask))
            data = {
                "img": data["img"],
                **meta,
                **gt,
                "ori_img": data["ori_img"],
                "distCoeffs": np.array(distCoeffs),
            }  # noqa
        data["img"] = data["img"].transpose(2, 0, 1)
        return data


@OBJECT_REGISTRY.register
class RoI3DDetInputPadding(DetInputPadding):
    def __call__(self, data):
        data = super().__call__(data)

        w_scale = 1.0 / data["trans_mat"][0, 0]
        h_scale = 1.0 / data["trans_mat"][1, 1]

        data["calib"][0, 2] += w_scale * self.input_padding[0]
        data["calib"][1, 2] += h_scale * self.input_padding[1]

        return data
