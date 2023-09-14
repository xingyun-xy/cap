import cv2
import numpy as np

from cap.registry import OBJECT_REGISTRY
from .affine import (
    AffineMat2DGenerator,
    ImageAffineTransform,
    get_affine_image_resize,
    resize_affine_mat,
)

__all__ = ["SemanticSegAffineAugTransformerEx"]


@OBJECT_REGISTRY.register
class SemanticSegAffineAugTransformerEx(object):
    """
    Multi scale affine transformer for semantic segmentation.

    Parameters
    ----------
    label_scales : iterable of float
        Output label scales
    resize_wh : list/tuple of 2 int, optional
        Resize input image to target size, by default None
    **kwargs :
        Please see :py:func:`get_affine_image_resize` and
        :py:class:`ImageAffineTransform`
    """

    def __init__(
        self,
        target_wh,
        label_scales,
        scale_type="W",
        inter_method=10,
        use_pyramid=False,
        pyramid_min_step=0.7,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
        center_aligned=False,
        rand_scale_range=(1.0, 1.0),
        rand_translation_ratio=0.0,
        rand_aspect_ratio=0,
        rand_rotation_angle=0,
        flip_prob=0,
        label_padding_value=-1,
        norm_wh=None,
        norm_scale=None,
        resize_wh=None,
        adapt_diff_resolution=False,
        padding_in_network_lrub=None,
    ):
        self._img_affine_ts = ImageAffineTransform(
            dst_wh=target_wh,
            inter_method=inter_method,
            border_value=0,
            use_pyramid=use_pyramid,
            pyramid_min_step=pyramid_min_step,
            pyramid_max_step=pyramid_max_step,
            pixel_center_aligned=pixel_center_aligned,
        )
        self._label_affine_ts = ImageAffineTransform(
            dst_wh=target_wh,
            inter_method=0,
            border_value=label_padding_value,
            use_pyramid=False,
            pyramid_min_step=pyramid_min_step,
            pyramid_max_step=pyramid_max_step,
            pixel_center_aligned=pixel_center_aligned,
        )
        if padding_in_network_lrub is not None:
            assert len(padding_in_network_lrub) == 4
            padded_target_w = (
                target_wh[0]
                + padding_in_network_lrub[0]
                + padding_in_network_lrub[1]
            )  # noqa
            padded_target_h = (
                target_wh[1]
                + padding_in_network_lrub[2]
                + padding_in_network_lrub[3]
            )  # noqa
            label_target_wh = [padded_target_w, padded_target_h]
            self.affine_matrix_for_padding = np.array(
                [
                    [1.0, 0, padding_in_network_lrub[0]],
                    [0, 1.0, padding_in_network_lrub[2]],
                ]
            )  # noqa
            self._label_affine_for_padding_ts = ImageAffineTransform(
                dst_wh=label_target_wh,
                inter_method=0,
                border_value=label_padding_value,
                use_pyramid=False,
                pyramid_min_step=pyramid_min_step,
                pyramid_max_step=pyramid_max_step,
                pixel_center_aligned=pixel_center_aligned,
            )
        else:
            self._label_affine_for_padding_ts = None
            label_target_wh = target_wh

        def _get_affine_mat(scale):
            linear_a = int(1.0 / scale)
            linear_b = int(linear_a / 2)
            output_wh = [
                int(label_target_wh[0] * scale),
                int(label_target_wh[1] * scale),
            ]

            ptr_before = np.array(
                [
                    [linear_b, linear_b],
                    [linear_b, output_wh[1] * linear_a + linear_b],
                    [
                        output_wh[1] * linear_a + linear_b,
                        output_wh[1] * linear_a + linear_b,
                    ],  # noqa
                ]
            )
            ptr_after = np.array(
                [[0, 0], [0, output_wh[1]], [output_wh[1], output_wh[1]]]
            )

            affine_mat = AffineMat2DGenerator.getAffineTransform(
                ptr_before, ptr_after
            )[0:2]
            return affine_mat

        self._label_aux_ts_affine_mat = [
            _get_affine_mat(scale_i) for scale_i in label_scales
        ]
        self._label_aux_ts = [
            ImageAffineTransform(
                dst_wh=[
                    int(label_target_wh[0] * scale_i),
                    int(label_target_wh[1] * scale_i),
                ],  # noqa
                inter_method=0,
                border_value=label_padding_value,
                use_pyramid=False,
                pyramid_min_step=pyramid_min_step,
                pyramid_max_step=pyramid_max_step,
                pixel_center_aligned=False,
            )
            for scale_i in label_scales
        ]
        self._affine_kwargs = {
            "target_wh": target_wh,
            "scale_type": scale_type,
            "center_aligned": center_aligned,
            "rand_scale_range": rand_scale_range,
            "rand_translation_ratio": rand_translation_ratio,
            "rand_aspect_ratio": rand_aspect_ratio,
            "rand_rotation_angle": rand_rotation_angle,
            "flip_prob": flip_prob,
            "norm_wh": norm_wh,
            "norm_scale": norm_scale,
        }
        self._resize_wh = resize_wh
        self._adapt_diff_resolution = adapt_diff_resolution

    def __call__(self, data):
        img = data["img"]
        label = data["anno"]
        aux_array = np.ones_like(label)
        label = np.where(label == 255, aux_array * -1, label)
        assert img.shape[:2] == label.shape[:2]
        if self._adapt_diff_resolution and self._resize_wh:

            orgin_wh = img.shape[:2][::-1]
            resize_wh_ratio = float(self._resize_wh[0]) / float(
                self._resize_wh[1]
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

        if self._resize_wh is None:
            affine_mat = AffineMat2DGenerator.identity()
            img_wh = img.shape[:2][::-1]
        else:
            affine_mat = resize_affine_mat(
                img.shape[:2][::-1], self._resize_wh
            )  # noqa
            img_wh = self._resize_wh

        affine_mat = AffineMat2DGenerator.stack_affine_transform(
            affine_mat,
            get_affine_image_resize(img_wh, **self._affine_kwargs).mat,
        )[:2]

        ts_img = self._img_affine_ts(img, affine_mat)

        ts_label = self._label_affine_ts(label, affine_mat)

        if self._label_affine_for_padding_ts is not None:
            ts_label = self._label_affine_for_padding_ts(
                ts_label, self.affine_matrix_for_padding
            )  # noqa

        ts_aux_labels = [
            aux_ts_i(ts_label, affine_mat_i)
            for aux_ts_i, affine_mat_i in zip(
                self._label_aux_ts, self._label_aux_ts_affine_mat
            )
        ]

        ts_aux_labels = [
            label_i.reshape((1, label_i.shape[0], label_i.shape[1]))
            for label_i in ts_aux_labels
        ]
        ts_aux_labels = [
            label_i.astype(np.float32, copy=False) for label_i in ts_aux_labels
        ]
        data = {
            "img": ts_img,
            "labels": ts_aux_labels,
        }
        data["img"] = data["img"].transpose(2, 0, 1)
        return data
