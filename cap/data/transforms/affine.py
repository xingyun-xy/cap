import math
import random
from collections import namedtuple

import cv2
import numpy as np

from cap.core.affine import (
    bbox_affine_transform,
    get_affine_transform,
    point_2D_affine_transform,
)
from cap.utils.apply_func import _as_list

__all__ = [
    "_pad_array",
    "resize_affine_mat",
    "AffineMat2DGenerator",
    "AffineMatFromROIBoxGenerator",
    "ImageAffineTransform",
    "AlphaImagePyramid",
    "LabelAffineTransform",
    "AffineAugMat",
    "Point2DAffineTransform",
    "image_transform",
    "get_affine_image_resize",
]

AffineAugMat = namedtuple("AffineAugMat", ["mat", "flipped"])

numeric_types = (float, int, np.generic)


def get_interp_method(interp, sizes=()):
    """Get the interpolation method for resize functions.

    The major purpose of this function is to wrap a random interp
    method selection and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look
        best with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer
        to http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width),
        if None provided, auto(9) will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError("Unknown interp method %d" % interp)
    return interp


def _pad_array(array, shape, name, pad_value=0):
    array_shape = array.shape
    assert len(array_shape) == len(
        shape
    ), "inconsistent shape dimension for %s, %s vs. %s" % (
        name,
        array_shape,
        shape,
    )
    new_shape = []
    for i, j in zip(array_shape, shape):
        j = i if j <= 0 else j
        assert (
            j >= i
        ), "%s with shape %s, some dims are larger than pad shape %s" % (
            name,
            array_shape,
            shape,
        )
        new_shape.append(j)
    new_shape = tuple(new_shape)
    if array_shape == new_shape:
        new_array = array
    else:
        new_array = np.ones(new_shape, dtype=array.dtype) * pad_value
        slices = tuple((slice(0, i) for i in array_shape))
        new_array[slices] = array
    return new_array


def get_affine_by_roi(
    src_wh,
    target_wh,
    src_roi,
    target_roi,
    center_aligned=True,
    rand_scale_range=(1, 1),
    rand_translation_ratio=0,
    rand_aspect_ratio=0,
    rand_rotation_angle=0,
    rand_shear_ratio=0,
    flip_prob=0,
):
    """Affine transformation mat by roi with augmentation.

    Args:
        src_wh : tuple of int,
            The size of input image
        target_wh : tuple of int
            The target size of output image
        src_roi : tuple or np.ndarray
            The source roi with layout of [(x0,y0), (x1,y1)]
            if provided as bbox,
            or [(x0,y0), (x1,y1), (x2,y2)] for triangle
        target_roi : tuple or np.ndarray
            The target roi with layout of [(x0,y0), (x1,y1)]
            if provided as bbox,
            or [(x0,y0), (x1,y1), (x2,y2)] for triangle
        center_aligned : bool, default=True
            Whether to align the image center during scaling transform.
            Note that this parameter only affect scaling from src_wh to
            to target_wh, and image will still be center misaligned due
            to translation that maximize the valid image area.
        rand_scale_range : tuple of float, default=(1, 1)
            The random scale to apply during scale translatoin.
        rand_translation_ratio : float, default=0
            The ratio of apply random translation after scaling.
            The translation will be caculated by :

            .. code-block:: python

                translation_x = target_wh[0] * np.random.uniform(
                    -1, 1) * rand_translation_ratio
                translation_y = target_wh[1] * np.random.uniform(
                    -1, 1) * rand_translation_ratio

        rand_aspect_ratio : float, default=0
            The ratio of random aspect augmentation. The valid value
            range will be 0-0.5. It will works in the following way:

            .. code-block:: python

                aspect = math.exp(random(-rand_aspect_ratio,
                        rand_aspect_ratio))
                x *= aspect
                y /= aspect

        rand_rotation_angle : int, default=0
            The max angle of rotation augmentation. Positive value means
            anti-clock rotation.
        flip_prob : float, default=0
            The probability of flipping image.

    Returns:
        affine_mat : :py:class:`AffineAugMat`
            The final affine transformation mat and other information.
            AffineAugMat.mat: The final affine mat of shape 2x3
            AffineAugMat.flipped: Whether to flip or not.
    """
    assert len(src_wh) == 2
    assert len(target_wh) == 2
    src_roi = np.array(src_roi)
    target_roi = np.array(target_roi)

    assert len(rand_scale_range) == 2
    assert rand_scale_range[0] <= rand_scale_range[1]
    assert rand_translation_ratio <= 0.5
    assert rand_translation_ratio >= 0
    assert rand_aspect_ratio >= 0
    assert rand_aspect_ratio <= 0.5
    assert rand_shear_ratio >= 0
    assert rand_shear_ratio <= 0.5
    assert flip_prob >= 0 and flip_prob <= 1

    rand_scale = np.random.uniform(rand_scale_range[0], rand_scale_range[1])
    translation_x = (
        target_wh[0] * np.random.uniform(-1, 1) * rand_translation_ratio
    )
    translation_y = (
        target_wh[1] * np.random.uniform(-1, 1) * rand_translation_ratio
    )
    shear_x = np.random.uniform(-1, 1) * rand_shear_ratio
    shear_y = np.random.uniform(-1, 1) * rand_shear_ratio
    angle = np.random.uniform(-1, 1) * rand_rotation_angle

    flip_flag = False
    flip_mat = AffineMat2DGenerator.identity()
    if flip_prob > 0 and flip_prob > np.random.uniform(0, 1):
        flip_flag = True
        flip_mat = AffineMat2DGenerator.flip_with_axis(
            center_xy=np.array(target_wh) / 2
        )

    def roi2ptr3(roi):
        # change roi format to [center, rb, rt] or keep origin 3ptr
        if roi.shape == (2, 2):
            roi = roi.reshape(4)
            return np.array(
                [
                    [(roi[0] + roi[2]) / 2, (roi[1] + roi[3]) / 2],
                    [roi[2], roi[3]],
                    [roi[2], roi[1]],
                ]
            )
        else:
            assert roi.shape == (3, 2)
            return roi

    source_ptr3 = roi2ptr3(src_roi)
    target_ptr3 = roi2ptr3(target_roi)
    ct_x, ct_y = target_ptr3[0, :].copy()

    stacked_mat = [
        AffineMat2DGenerator.getAffineTransform(source_ptr3, target_ptr3),
        AffineMat2DGenerator.translation(-ct_x, -ct_y),
        AffineMat2DGenerator.aspect_ratio_aug(rand_aspect_ratio),
        AffineMat2DGenerator.scale(rand_scale, rand_scale),
        AffineMat2DGenerator.shear((0, 0), shear_x, shear_y),
        AffineMat2DGenerator.translation(ct_x, ct_y),
        AffineMat2DGenerator.rotate(
            (target_wh[0] / 2, target_wh[1] / 2), angle=angle
        ),
        flip_mat,
    ]

    stacked_mat = AffineMat2DGenerator.stack_affine_transform(*stacked_mat)
    src_border = np.array([(0, 0), (src_wh[0], 0), src_wh, (0, src_wh[1])])
    new_src_border = point_2D_affine_transform(src_border, stacked_mat[0:2])

    def get_offset(a2, a1):
        if a2 >= 0 and a1 >= 0:
            return min(a2, a1)
        elif a2 >= 0 and a1 <= 0:
            return np.random.uniform(a1, a2)
        elif a2 < 0 and a1 < 0:
            return max(a2, a1)
        else:
            return 0

    if not center_aligned:
        x_min = np.min(new_src_border[:, 0])
        y_min = np.min(new_src_border[:, 1])
        x_max = np.max(new_src_border[:, 0])
        y_max = np.max(new_src_border[:, 1])

        x_dist2 = target_wh[0] - x_max
        x_dist1 = 0 - x_min
        y_dist2 = target_wh[1] - y_max
        y_dist1 = 0 - y_min
        translation_x += get_offset(x_dist2, x_dist1)
        translation_y += get_offset(y_dist2, y_dist1)

    stacked_mat = [
        stacked_mat,
        AffineMat2DGenerator.translation(translation_x, translation_y),
    ]

    return AffineAugMat(
        mat=AffineMat2DGenerator.stack_affine_transform(*stacked_mat)[0:2],
        flipped=flip_flag,
    )


def get_affine_roi_center_scale(
    src_wh, target_wh, roi_center_xy, roi_scale_xy, **kwargs
):
    """Affine transformation mat with augmentation given roi center and scale.

    Args:
        src_wh : tuple of int,
            The size of input image
        target_wh : tuple of int
            The target size of output image
        roi_center_xy : tuple of (x, y)
            The roi center in the original image space. It will be the
            center in the target image if `center_aligned` is True and
            no additional random augmentation is set.
        roi_scale_xy : tuple of (x, y)
            The scale of roi after transform.
        **kwargs : Please see :py:meth:`get_affine_by_roi` and

    Returns:
        affine_mat : :py:class:`AffineAugMat`
            The final affine transformation mat and other information.
            AffineAugMat.mat: The final affine mat of shape 2x3
            AffineAugMat.flipped: Whether to flip or not.
    """
    assert len(src_wh) == 2
    assert len(target_wh) == 2
    assert len(roi_center_xy) == 2
    assert len(roi_scale_xy) == 2
    ori_ptrs = np.array([roi_center_xy, roi_center_xy, roi_center_xy])
    ori_ptrs[1, 0] += 1
    ori_ptrs[2, 1] += 1

    new_ct = np.array(target_wh) / 2.0
    new_ptrs = np.array([new_ct, new_ct, new_ct])
    new_ptrs[1, 0] += roi_scale_xy[0]
    new_ptrs[2, 1] += roi_scale_xy[1]

    return get_affine_by_roi(src_wh, target_wh, ori_ptrs, new_ptrs, **kwargs)


def resize_affine_mat(src_wh, dst_wh):
    """Get the affine matrix for resize.

    Args:
        src_wh: tuple like (w, h). The source width and height of image
        dst_wh: tuple like (w, h). The target width and height of image
    """
    assert len(src_wh) == 2
    assert len(dst_wh) == 2
    ptr_before = np.array([[0, 0], [src_wh[0], 0], [0, src_wh[1]]])
    ptr_after = np.array([[0, 0], [dst_wh[0], 0], [0, dst_wh[1]]])
    return AffineMat2DGenerator.getAffineTransform(ptr_before, ptr_after)[0:2]


class AffineMat2DGenerator(object):
    """Generate affine transformation matrix.

    All return matrix is in the shape of 3x3

    """

    @staticmethod
    def identity():
        """Return the 3x3 identity mat."""
        return np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    @staticmethod
    def translation(x, y):
        """Return the translation matrix of affine.

        Args:
            x: int or float. The x axis.
                Positive value means moving image right.
            y: int or float. The y axis.
                Positive value means moving image down.
        """
        assert isinstance(x, numeric_types)
        assert isinstance(y, numeric_types)
        return np.array([[1.0, 0, x], [0, 1.0, y], [0, 0, 1.0]])

    @staticmethod
    def scale(x, y):
        """Return the scaling matrix of affine.

        Args:
            x: int or float. The scale of x axis.
            y: int or float. The scale of y axis.
        """
        assert isinstance(x, numeric_types)
        assert isinstance(y, numeric_types)
        return np.array([[x, 0, 0], [0, y, 0], [0, 0, 1.0]])

    @staticmethod
    def flip_with_axis(center_xy, with_x=True, with_y=False):
        """Return the flipping matrix of affine.

        Args:
            center_xy: list/tuple of int. The center to perform flipping.
            with_x: bool. Default=True. Whether to flip along x axis.
            with_y: bool. Default=False. Whether to flip along y axis.
        """
        ret = AffineMat2DGenerator.identity()
        if with_x:
            ret[0, 0] = -1
        if with_y:
            ret[1, 1] = -1
        ret = np.dot(
            ret, AffineMat2DGenerator.translation(-center_xy[0], -center_xy[1])
        )
        ret = np.dot(
            AffineMat2DGenerator.translation(center_xy[0], center_xy[1]), ret
        )
        return ret

    @staticmethod
    def rotate(center_xy, angle, scale=1):
        """Return the rotation matrix of affine. The same as cv2.getRotationMatrix2D.

        Args:
            center_xy: tuple of format(x, y). The rotation center
            angle: numeric_types. The angle of rotation.
                Positive value means anti-clock rotation.
            scale: numeric_types. Default=1. Isotropic scale factor.
        """
        M = cv2.getRotationMatrix2D(center_xy, angle, scale)
        return M

    @staticmethod
    def shear(center_xy, s_x, s_y):
        """Return the affine matrix by shear.

        Args:
            center_xy: tuple of format(x, y). The center
            s_x: numeric_types. The shear ratio along x axis
            s_y: numeric_types. The shear ratio along y axis
        """
        ret = np.array([[1.0, s_x, 0], [s_y, 1.0, 0], [0, 0, 1.0]])
        ret = np.dot(
            ret, AffineMat2DGenerator.translation(-center_xy[0], -center_xy[1])
        )
        return np.dot(
            AffineMat2DGenerator.translation(center_xy[0], center_xy[1]), ret
        )

    @staticmethod
    def aspect_ratio_aug(max_x_ratio=0):
        """Aspect ratio augmentation.

        Return the affine matrix that apply random.
        aspect ratio augmentation on image.

        Args:
            max_x_ratio: numeric_types, default=0.
                The max augumanted aspect ratio applied to x axis.
                The actual x ratio will be calculated by
                `math.exp(np.random.uniform(-max_x_ratio, max_x_ratio))`
        """
        x_ratio = math.exp(np.random.uniform(-max_x_ratio, max_x_ratio))
        return AffineMat2DGenerator.scale(x_ratio, 1.0 / x_ratio)

    @staticmethod
    def resize_by_min_edge(src_wh, target_min_edge):
        """Scale the image with the target min edge.

        Return the affine matrix that scale the image that
        the target min edge is in certain length.

        Args:
            src_wh: tuple of format(w, h). The source image width and height
            target_min_edge: numeric_types. The target minimal edge length
        """
        assert len(src_wh) == 2
        min_edge = min(src_wh[0], src_wh[1])
        target_scale = target_min_edge / float(min_edge)
        return AffineMat2DGenerator.scale(target_scale, target_scale)

    @staticmethod
    def resize_by_min_max(src_wh, min_edge, max_edge):
        """Scale the image like Faster RCNN.

        Return the affine matrix that scale the image that
        the target min edge is equal to `min_edge` if the max edge
        does not exceed `max_edge`, otherwise the target max edge is
        resized to `max_edge`. This setting is used widely used in
        Faster RCNN

        Args:
            src_wh: tuple of format(w, h). The source image width and height
            min_edge: numeric_types. The target min edge length
            max_edge: numeric_types. The target max edge length
        """
        assert min_edge <= max_edge and min_edge > 0 and max_edge > 0
        assert len(src_wh) == 2
        im_max_edge = max(src_wh[0], src_wh[1])
        im_min_edge = min(src_wh[0], src_wh[1])

        target_scale = float(min_edge) / float(im_min_edge)
        if np.round(target_scale * im_max_edge) > max_edge:
            target_scale = float(np.floor(max_edge)) / float(im_max_edge)
        return AffineMat2DGenerator.scale(target_scale, target_scale)

    @staticmethod
    def resize_by_max_edge(src_wh, target_max_edge):
        """Scale the image with the target max edge.

        Return the affine matrix that scale the image that
        the target max edge is in certain length.

        Args:
            src_wh: tuple of format(w, h). The source image width and height
            target_max_edge: numeric_types. The target max edge length
        """
        assert len(src_wh) == 2
        max_edge = max(src_wh[0], src_wh[1])
        target_scale = target_max_edge / float(max_edge)
        return AffineMat2DGenerator.scale(target_scale, target_scale)

    @staticmethod
    def getAffineTransform(pts_before, pts_after):
        """Calculate an affine transform from three pairs of the corresponding points.

        Args:
            pts_before: np.ndarray. The points before transformation.
                It should be in the shape of (3, 2)
            pts_before: np.ndarray. The points after transformation.
                It should be in the shape of (3, 2)
        """
        assert isinstance(pts_before, np.ndarray)
        assert isinstance(pts_after, np.ndarray)
        assert pts_before.shape == (3, 2)
        assert pts_after.shape == (3, 2)
        M = cv2.getAffineTransform(
            pts_before.astype(np.float32), pts_after.astype(np.float32)
        )
        return M

    @staticmethod
    def stack_affine_transform(*mats):
        """Merge affine transform matrixes into a new one.

        The transform matrix will be applied one by one.

        Args:
            mats: list of np.ndarray.
                Affine transform with shape (2, 3) or (3, 3)

        Returns:
            merged_mat: np.ndarray.
                Merged affine transform with shape (2, 3)
        """
        if not mats:
            return AffineMat2DGenerator.identity()
        for mat in mats:
            assert isinstance(mat, np.ndarray), "type: {}".format(type(mat))

        merged = AffineMat2DGenerator.to3x3(mats[0])
        for mat in mats[1:]:
            mat = AffineMat2DGenerator.to3x3(mat)
            merged = np.dot(mat, merged)
        return merged

    @staticmethod
    def to3x3(mat):
        if mat.shape == (2, 3):
            new_mat = AffineMat2DGenerator.identity()
            new_mat[0:2] = mat[:]
            return new_mat
        assert mat.shape == (3, 3)
        return mat


class AffineMatFromROIBoxGenerator(object):
    """An generator that generate affine matrix for each roi.

    Args:
        target_wh : tuple/list of 2 int
            Target width and height
        scale_range : tuple/list of 2 float
            The minimum and maximum sampling scale.
        min_sample_num : int, optional
            The minimum sample number, by default 1
        max_sample_num : int, optional
            The maximum sample number, by default 5
        min_valid_edge : float, optional
            The minimum valid edge size, by default 1
        min_valid_area : float, optional
            The minimum valid bbox area, by default 8
        rand_sampling_bbox : bool, optional
            Whether randomly sample bbox, by default True
        reselect_ratio : bool, optional
            The probability to reselect a roi, by default 0
        kwargs : dict
            Please see :py:meth:`get_affine_roi_center_scale`
    """

    def __init__(
        self,
        target_wh,
        scale_range=(1.0 / 32, 4.0),
        min_sample_num=1,
        max_sample_num=5,
        min_valid_edge=1,
        min_valid_area=8,
        rand_sampling_bbox=True,
        reselect_ratio=0,
        **kwargs
    ):

        assert len(scale_range) == 2
        assert scale_range[0] <= scale_range[1]
        assert len(target_wh) == 2
        self._target_wh = target_wh
        self._min_scale = scale_range[0]
        self._max_scale = scale_range[1]
        self._min_sample_num = min_sample_num
        self._max_sample_num = max_sample_num
        self._kwargs = kwargs
        self._min_valid_edge = min_valid_edge
        self._min_valid_area = min_valid_area
        self._rand_sampling_bbox = rand_sampling_bbox
        self._reselect_ratio = reselect_ratio
        self._bbox_ts = LabelAffineTransform(label_type="box")

    def _is_bbox_valid_sample(self, bbox):
        centers = (bbox[:2] + bbox[2:4]) / 2
        center_valid = np.logical_and(
            (0, 0) <= centers, centers < self._target_wh
        ).all()

        lt_point = np.maximum(bbox[:2], (0, 0))
        rb_point = np.minimum(bbox[2:4], self._target_wh)
        box_size_valid = (lt_point + self._min_valid_edge < rb_point).all()

        area = (rb_point[0] - lt_point[0]) * (rb_point[1] - lt_point[1])
        area_valid = (area >= self._min_valid_area).all()

        return center_valid and box_size_valid and area_valid

    def _get_target_scale(self, bbox):
        centers = (bbox[:2] + bbox[2:4]) / 2
        s = np.random.uniform(self._min_scale, self._max_scale)
        return centers, s, s

    def __call__(self, bboxes, img_wh):
        assert isinstance(bboxes, np.ndarray)
        assert bboxes.ndim == 2
        assert len(img_wh) == 2

        def update_bbox_as_pos_status(bbox_as_pos_list, new_bbox):
            assert len(bbox_as_pos_list) == len(new_bbox)
            for i, box in enumerate(new_bbox):
                if bbox_as_pos_list[i]:
                    continue
                bbox_as_pos_list[i] = self._is_bbox_valid_sample(box)

        n_bbox = bboxes.shape[0]
        if self._rand_sampling_bbox:
            bbox_idx_list = np.random.permutation(n_bbox)
        else:
            bbox_idx_list = np.arange(n_bbox)
        bbox_as_positive = [False] * n_bbox
        cnt = 0
        for bbox_id in bbox_idx_list:
            re_select = np.random.uniform(0, 1) <= self._reselect_ratio
            if bbox_as_positive[bbox_id] and (not re_select):
                continue
            center, sx, sy = self._get_target_scale(bboxes[bbox_id])

            affine_aug_param = get_affine_roi_center_scale(
                img_wh,
                self._target_wh,
                roi_center_xy=center,
                roi_scale_xy=(sx, sy),
                **(self._kwargs)
            )
            # TODO(): bbox_label is useless here..
            bbox_label = self._bbox_ts(
                bboxes, affine_aug_param.mat, flip=affine_aug_param.flipped
            )
            update_bbox_as_pos_status(bbox_as_positive, bbox_label)
            yield affine_aug_param
            cnt += 1
            if cnt >= self._max_sample_num:
                break
        # random roi if no available bbox to explore
        if cnt < self._min_sample_num:
            for _ in range(self._min_sample_num - cnt):
                x = np.random.uniform(1, img_wh[0] - 1)
                y = np.random.uniform(1, img_wh[1] - 1)
                s = np.random.uniform(self._min_scale, self._max_scale)
                affine_aug_param = get_affine_roi_center_scale(
                    img_wh,
                    self._target_wh,
                    roi_center_xy=(x, y),
                    roi_scale_xy=(s, s),
                    **(self._kwargs)
                )
                bbox_label = self._bbox_ts(
                    bboxes, affine_aug_param.mat, flip=affine_aug_param.flipped
                )
                yield affine_aug_param


class ImageAffineTransform(object):
    """Apply affine transformation on image.

    Note that image should be in the shape of (H, W, C) or (H, W).

    Args:
        dst_wh: tuple like (w, h).
            The target width and height of transformed image.
        inter_method: int. Default=1
            Interpolation method used for resizing the image.

            Possible values:
            0: Nearest Neighbors Interpolation.
            1: Bilinear interpolation.
            2: Area-based (resampling using pixel area relation). It may be a
            preferred method for image decimation, as it gives moire-free
            results. But when the image is zoomed, it is similar to the Nearest
            Neighbors method. (used by default).
            3: Bicubic interpolation over 4x4 pixel neighborhood.
            4: Lanczos interpolation over 8x8 pixel neighborhood.
            9: Cubic for enlarge, area for shrink, bilinear for others
            10: Random select from interpolation method metioned above.
            Note:
            When shrinking an image, it will generally look best with
            AREA-based interpolation, whereas, when enlarging an image,
            it will generally look best with Bicubic (slow) or Bilinear
            (faster but still looks OK). More details can be found in
            the documentation of OpenCV, please refer to
            http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
        border_value: int
            Value used in case of a constant border; by default, it is 0.
        use_pyramid: bool. Default=False
            Whether to use image pyramid.
        pyramid_min_step: float, default=0.45
            The pyramid step to build. The step will be chosen from
            [min_step, max_step]
        pyramid_max_step: float, default=0.8
            The pyramid step to build. The step will be chosen from
            [min_step, max_step]
        pixel_center_aligned: bool, default=True
            Whether to use pixel center aligned version for affine
            transform.
    """

    def __init__(
        self,
        dst_wh,
        inter_method=1,
        border_value=0,
        use_pyramid=False,
        pyramid_min_step=0.45,
        pyramid_max_step=0.8,
        pixel_center_aligned=True,
    ):
        super(ImageAffineTransform, self).__init__()
        self._inter_method = inter_method
        self._border_value = border_value
        self._dst_wh = (int(dst_wh[0]), int(dst_wh[1]))
        self._use_pyramid = use_pyramid
        self._pyramid_min_step = pyramid_min_step
        self._pyramid_max_step = pyramid_max_step
        self._pixel_center_aligned = pixel_center_aligned

    def __call__(self, image, affine_mat):
        """Transform image by affine matrix.

        Note that affine matrix should be in the shape of (2, 3).

        Args:
            image: NDArray or np.ndarray, or AlphaImagePyramid.
                Image to apply transformation. It should be in the shape of
                (H, W, C) or (H, W).
            affine_mat: np.ndarray. The affine matrix.
                It should be in the shape of (2, 3).

        Returns:
            ret: NDArray. The image after transformation.
        """
        # if isinstance(image, NDArray):
        #     image = image.asnumpy()
        assert isinstance(image, (np.ndarray, AlphaImagePyramid))
        is_pyramid_image = isinstance(image, AlphaImagePyramid)
        ori_image = image.image if is_pyramid_image else image

        assert len(image.shape) >= 2
        assert len(image.shape) <= 3
        assert isinstance(affine_mat, np.ndarray)

        if self._pixel_center_aligned:
            affine_mat = AffineMat2DGenerator.stack_affine_transform(
                AffineMat2DGenerator.translation(0.5, 0.5),
                affine_mat,
                AffineMat2DGenerator.translation(-0.5, -0.5),
            )

        if affine_mat.shape == (3, 3):
            affine_mat = affine_mat[0:2]
        assert affine_mat.shape == (2, 3)
        h, w = image.shape[0:2]
        inter_method = get_interp_method(
            self._inter_method, (h, w, self._dst_wh[1], self._dst_wh[0])
        )
        # inter_method = random.randint(0, 4)  # _inter_method=10

        border_value = self._border_value
        img_c = 1
        if len(image.shape) == 3:
            img_c = image.shape[2]
        if isinstance(border_value, numeric_types):
            border_value = (border_value,) * img_c
        else:
            assert len(self._border_value) == img_c, "{} != {}".format(
                len(self._border_value), img_c
            )
        border_value = [float(v) for v in border_value]
        if len(border_value) == 1:
            border_value = border_value[0]
        new_affine_mat = affine_mat.astype(np.float)

        if self._use_pyramid or is_pyramid_image:
            # if use pyramid, we have to get the scale,
            # and the new affine matrix
            if is_pyramid_image:
                pyramid = image
            else:
                pyramid = AlphaImagePyramid(
                    image,
                    scale_step=np.random.uniform(
                        self._pyramid_min_step, self._pyramid_max_step
                    ),
                )
            cur_scale = 1.0
            for _ in range(pyramid.max_layer_num - 1):
                dist = (
                    new_affine_mat[0, 0] ** 2 + new_affine_mat[1, 0] ** 2,
                    new_affine_mat[0, 1] ** 2 + new_affine_mat[1, 1] ** 2,
                )
                if (dist[0] >= pyramid.scale_step ** 2) or (
                    dist[1] >= pyramid.scale_step ** 2
                ):
                    break
                cur_scale *= pyramid.scale_step
                new_affine_mat[:, 0:2] /= pyramid.scale_step
            if cur_scale < 1:
                pyramid.build_from(cur_scale, inter_method=inter_method)
                new_img, tmp_scale, affine_inv = pyramid.get_layer_by_scale(
                    cur_scale
                )

                new_affine_mat = np.dot(
                    AffineMat2DGenerator.to3x3(affine_mat.astype(np.float)),
                    affine_inv,
                )
                ret = cv2.warpAffine(
                    new_img,
                    new_affine_mat[0:2],
                    self._dst_wh,
                    flags=inter_method,
                    borderValue=border_value,
                )
            else:
                ret = cv2.warpAffine(
                    ori_image,
                    affine_mat.astype(np.float),
                    self._dst_wh,
                    flags=inter_method,
                    borderValue=border_value,
                )
        else:
            ret = cv2.warpAffine(
                ori_image,
                affine_mat.astype(np.float),
                self._dst_wh,
                flags=inter_method,
                borderValue=border_value,
            )
        return ret


class AlphaImagePyramid(object):
    """Pyramid for image resize.

    The image pyramid will cache all layers until
    you call ``clear`` to clean all cache.

    Args:
        image: np.ndarray. The image to build.
        scale_step: float. Default=0.8.
            The scale step for building image pyramid
        max_layer_num: int. Default=30.
            The maximum number of layers for image pyramid
    """

    def __init__(self, image, scale_step=0.8, max_layer_num=30):
        assert scale_step < 1
        assert scale_step > 0.1
        assert max_layer_num > 0

        self._scale_step = scale_step
        self._max_layer_num = max_layer_num

        self._images_cached = []
        self._scales = []
        self._layer_acc_affine_inv = []

        # if isinstance(image, NDArray):
        #     image = image.asnumpy()
        # else:
        assert isinstance(image, np.ndarray)
        # img_wh = (image.shape[1], image.shape[0])
        cur_affine_inv = AffineMat2DGenerator.identity()
        self._layer_acc_affine_inv.append(cur_affine_inv)
        self._images_cached.append(image)
        self._scales.append(1.0)

    @property
    def scale_step(self):
        return self._scale_step

    @property
    def max_layer_num(self):
        return self._max_layer_num

    @property
    def layer_num(self):
        return len(self._scales)

    @property
    def layer_acc_affine_inv(self):
        return self._layer_acc_affine_inv

    @property
    def image(self):
        return self._images_cached[0]

    @property
    def shape(self):
        return self.image.shape

    def build_from(self, scale, inter_method=1):
        """Build image pyramid given image and target scale.

        Args:
            scale: float.
                The target scale that suggest the number of layer to build.
            inter_method: int. Default=1.
                interpolation method for all resizing operations

                Possible values:
                0: Nearest Neighbors Interpolation.
                1: Bilinear interpolation.
                2: Area-based (resampling using pixel area relation).
                It may be a preferred method for image decimation,
                as it gives moire-free results. But when the image is zoomed,
                it is similar to the Nearest Neighbors method.
                (used by default).
                3: Bicubic interpolation over 4x4 pixel neighborhood.
                4: Lanczos interpolation over 8x8 pixel neighborhood.
        """
        assert scale > 0
        image = self.image
        img_wh = (image.shape[1], image.shape[0])
        min_build_scale = self._scales[-1]
        while min_build_scale > scale:
            min_build_scale *= self._scale_step
            dst_wh = (img_wh[0] * min_build_scale, img_wh[1] * min_build_scale)
            if dst_wh[0] < 8 or dst_wh[1] < 8:
                break
            img = cv2.resize(
                self._images_cached[-1],
                dsize=(int(dst_wh[0]), int(dst_wh[1])),
                interpolation=inter_method,
            )
            last_shape = self._images_cached[-1].shape
            last_wh = (last_shape[1], last_shape[0])
            cur_affine_inv = self._get_resize_affine_inv(
                int(dst_wh[0]) / float(last_wh[0]),
                int(dst_wh[1]) / float(last_wh[1]),
            )

            self._scales.append(min_build_scale)
            self._images_cached.append(img)
            self._layer_acc_affine_inv.append(
                np.dot(self._layer_acc_affine_inv[-1], cur_affine_inv)
            )
            if len(self._scales) >= self._max_layer_num:
                print("Build too much layers! Please use large scale_step")
                break
        assert len(self._scales) == len(self._images_cached)
        assert len(self._scales) == len(self._layer_acc_affine_inv)

    def _get_resize_affine_inv(self, scale_x, scale_y):
        return np.array(
            [
                [1.0 / scale_x, 0, -(scale_x - 1) / (2 * scale_x)],
                [0, 1.0 / scale_y, -(scale_y - 1) / (2 * scale_y)],
                [0, 0, 1.0],
            ]
        )

    def get_layer_by_scale(self, scale):
        """Given a scale.

        Given a scale, get the smallest layer whose scale is larger than
        target.

        Args:
            scale: float. The scale to query.

        Returns:
            np.ndarray. The image from a layer of pyramid
            float. The scale of returned image

        """
        assert len(self._scales) >= 1
        # err = abs(scale - self._scales[0])
        idx = 0
        for i, cached_scale in enumerate(self._scales):
            if cached_scale > scale:
                idx = max(0, i - 1)

        return (
            self._images_cached[idx],
            self._scales[idx],
            self._layer_acc_affine_inv[idx],
        )


class SwapPoints(object):
    """Swap points.

    Note that the points data should be in the shape of
    (I, N, D), where I is the number of instance, N is the number of points,
    D is the dimension of each point.

    Args:
        swap_pairs: list/tuple of list/tuple of 2 int Swap point ids.
    """

    def __init__(self, swap_pairs):
        swap_pairs = self._check_and_convert_swap_pairs(swap_pairs)
        self._ptr1 = np.array([pair[0] for pair in swap_pairs])
        self._ptr2 = np.array([pair[1] for pair in swap_pairs])
        self._max_ptr = max([max(pair) for pair in swap_pairs])

    def _check_and_convert_swap_pairs(self, swap_pairs):
        assert isinstance(swap_pairs, (list, tuple))
        for pair in swap_pairs:
            assert isinstance(pair, (list, tuple))
            assert len(pair) == 2
            for ptr in pair:
                assert isinstance(ptr, int) and ptr >= 0
        swap_pairs = [sorted(pair) for pair in swap_pairs]
        new_swap_pairs = []
        for pair in swap_pairs:
            if pair not in new_swap_pairs:
                new_swap_pairs.append(pair)
        num_pairs = len(new_swap_pairs)
        for i in range(num_pairs):
            for j in range(i + 1, num_pairs):
                assert new_swap_pairs[i][0] not in new_swap_pairs[j], (
                    "ptr %d duplicate occurs" % new_swap_pairs[i][0]
                )
                assert new_swap_pairs[i][1] not in new_swap_pairs[j], (
                    "ptr %d duplicate occurs" % new_swap_pairs[i][1]
                )
                # remove identity mapping
                new_swap_pairs = list(
                    filter(lambda x: x[0] != x[1], new_swap_pairs)
                )
        return new_swap_pairs

    def __call__(self, data):
        """Swap points.

        Args:
            data: np.ndarray. Points to apply transformation.
                It should be in the shape of (I, N, D).

        Returns:
            new_data: np.ndarray. Points after transformation.
        """
        assert isinstance(data, np.ndarray)
        ori_shape = data.shape
        if data.ndim == 1:
            data = data.reshape((1, 1, data.shape[0]))
        elif data.ndim == 2:
            data = data.reshape((1, data.shape[0], data.shape[1]))
        else:
            assert data.ndim == 3
        assert (
            data.shape[1] > self._max_ptr
        ), "Given %d points, but the maximum point id in swap_pairs is %d" % (
            data.shape[1],
            self._max_ptr,
        )
        new_data = data.copy()
        new_data[:, self._ptr1, :] = data[:, self._ptr2, :]
        new_data[:, self._ptr2, :] = data[:, self._ptr1, :]
        # reshape back to original shape
        new_data = new_data.reshape(ori_shape)
        return new_data


class LabelAffineTransform(object):
    # TODO(): Very very inefficient implementation!!!
    def __init__(self, label_type, point_id_swap_pairs=None):
        assert label_type in ["box", "point"]
        self._label_type = label_type
        if label_type == "box":
            self._ts = bbox_affine_transform
        elif label_type == "point":
            self._ts = point_2D_affine_transform
            assert (
                point_id_swap_pairs is not None
            ), "You have to provide point_id_swap_pairs for points affine"
            self._ptr_swap = SwapPoints(point_id_swap_pairs)

    def __call__(self, data, affine_mat, flip=False):
        ret_data = [
            self._ts(i, affine_mat) if i.size > 0 else i
            for i in _as_list(data)
        ]
        if self._label_type == "point" and flip is True:
            ret_data = [
                self._ptr_swap(i) if i.size > 0 else i for i in ret_data
            ]
        if len(ret_data) == 1:
            ret_data = ret_data[0]
        return ret_data


class Point2DAffineTransform(object):
    """Apply an affine transformation on 2D points.

    Note that the latest dimension of 2D points data should be 2.
    Data should be in (x, y) format.

    """

    def __init__(self):
        super(Point2DAffineTransform, self).__init__()

    def __call__(self, data, affine_mat):
        """Transform 2D points by affine matrix.

        Note that affine matrix should be in the shape of (2, 3),
        and the last dimension of 2D points data should be 2.

        Args:
            data: np.ndarray.
                2D Points to apply transformation.
                The last dimension should be 2.
            affine_mat: np.ndarray.
                The affine matrix. It should be in the shape of (2, 3).

        Returns:
            new_data: np.ndarray
                The 2D points after transformation.
        """
        assert isinstance(data, np.ndarray)
        if affine_mat.shape == (3, 3):
            affine_mat = affine_mat[0:2]
        assert affine_mat.shape == (2, 3)
        assert data.shape[-1] == 2, data.shape
        ori_shape = data.shape
        data = data.reshape((-1, 2))
        new_data = np.ones(
            (data.shape[0], data.shape[1] + 1), dtype=data.dtype
        )
        new_data[:, 0:2] = data
        new_data = np.dot(new_data, np.transpose(affine_mat))
        return new_data.reshape(ori_shape)


def image_transform(img, input_wh, keep_res, shift):
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
    if keep_res:
        size = np.array(input_wh, dtype=np.int32)
    else:
        size = np.array([width, height], dtype=np.int32)

    # trans_input = get_affine_transform(center, size, 0, input_wh, shift=shift)  # noqa
    trans_input = get_affine_transform(
        size=size, rotation=0, out_size=input_wh, center_shift=size * shift
    )  # noqa

    inp = cv2.warpAffine(
        img, trans_input, tuple(input_wh), flags=cv2.INTER_LINEAR
    )

    trans_matrix = {"center": center, "size": size, "trans_input": trans_input}
    return inp, trans_matrix


def rand_scale_affine_mat(
    src_wh, dst_wh=None, min_scale=1.0, max_scale=1.0, align_type="rand"
):
    """Get the affine matrix for randomly chose a scale in given range.

    If the actually resized is smaller or larger than given target size,
    cropping is based on the align_type.  This transform will not change
    the aspect ratio of image.

    Args:
        src_wh: tuple like (w, h).
            The source width and height of image
        dst_wh: tuple like (w, h).
            The target width and height of image.
            It is used for ``warpAffine``.
            The actual image size is determined by the random scale.
        min_scale: float. Default=1.0.
            The minimal scale.
        max_scale: float. Default=1.0.
            The maximum scale
        align_type: str. Default="rand".
            The type of alignment for cropping or padding.

            Possible values:
            1. "rand": random cropping or padding.
            2. "center": force image center in the target.
            3. "left-top": force the left top corner is aligned.
    """

    assert min_scale <= max_scale
    assert len(src_wh) == 2
    if dst_wh is None:
        dst_wh = src_wh
    assert len(dst_wh) == 2
    dst_wh = np.array(dst_wh).astype(np.int32)
    tar_wh = np.array(src_wh, dtype=np.float) * np.random.uniform(
        min_scale, max_scale
    )
    tar_wh = tar_wh.astype(np.int32)
    if align_type == "left-top":
        # use left-top, right-top, left-bottom as reference points
        ptr_before = np.array([[0, 0], [src_wh[0], 0], [0, src_wh[1]]])
        ptr_after = np.array([[0, 0], [tar_wh[0], 0], [0, tar_wh[1]]])
    elif align_type == "center":
        # use left-top, right-top, center as reference points
        ptr_before = np.array(
            [[0, 0], [src_wh[0], 0], [src_wh[0] / 2.0, src_wh[1] / 2.0]]
        )
        ptr_after = np.array(
            [
                [
                    dst_wh[0] / 2.0 - tar_wh[0] / 2.0,
                    dst_wh[1] / 2.0 - tar_wh[1] / 2.0,
                ],
                [
                    dst_wh[0] / 2.0 + tar_wh[0] / 2.0,
                    dst_wh[1] / 2.0 - tar_wh[1] / 2.0,
                ],
                [dst_wh[0] / 2.0, dst_wh[1] / 2.0],
            ]
        )
    elif align_type == "rand":
        # use left-top, right-top, right-bottom as reference points
        ptr_before = np.array([[0, 0], [src_wh[0], 0], [src_wh[0], src_wh[1]]])
        ptr_after = np.array([[0, 0], [tar_wh[0], 0], [tar_wh[0], tar_wh[1]]])
        shift = dst_wh - tar_wh

        def get_shift(shift_v):
            if shift_v > 0:
                return np.random.randint(0, shift_v + 1)
            else:
                return np.random.randint(shift_v, 1)

        shift1 = get_shift(shift[0])
        shift2 = get_shift(shift[1])
        shift_xy = np.array([shift1, shift2])

        ptr_after = ptr_after + np.broadcast_to(shift_xy, ptr_after.shape)

    else:
        raise NotImplementedError(
            "not supported align_type: {}".format(align_type)
        )
    return AffineMat2DGenerator.getAffineTransform(ptr_before, ptr_after)[0:2]


def get_affine_image_resize(
    src_wh,
    target_wh,
    scale_type="W",
    center_aligned=True,
    norm_wh=None,
    norm_scale=None,
    rand_scale_range=(1, 1),
    **kwargs
):
    """Affine transformation matrix with augmentation for whole image.

    Args:
        src_wh : tuple of int,
            The size of input image
        target_wh : tuple of int
            The target size of output image
        scale_type : str, default="W"
            The way to determin scale. Possivle value:
            "W": scale = w_scale = float(target_wh[0])/img_wh[0]
            "H": scale = h_scale = float(target_wh[1])/img_wh[1]
            "MIN": scale = min(w_scale, h_scale)
            "MAX": scale = max(w_scale, h_scale)
        center_aligned : bool, default=True
            Whether to align the image center during scaling transform.
            Note that this parameter only affect scaling from src_wh to
            to target_wh, and image will still be center misaligned due
            to random translation.
        rand_scale_range : tuple of float, default=(1, 1)
            The random scale to apply during scale translatoin.
        **kwargs :
            Please see :py:meth:`get_affine_by_roi` and

    Returns:
        affine_mat : :py:class:`AffineAugMat`
            The final affine transformation mat and other information.
            AffineAugMat.mat: The final affine mat of shape 2x3
            AffineAugMat.flipped: Whether to flip or not.
    """
    if norm_wh is None:
        norm_wh = target_wh

    assert len(src_wh) == 2
    assert len(target_wh) == 2
    assert len(norm_wh) == 2
    assert scale_type in ["W", "H", "MIN", "MAX"]
    assert len(rand_scale_range) == 2
    assert rand_scale_range[0] <= rand_scale_range[1]

    def get_scale(img_wh, target_wh, scale_type="W"):
        """Faster-RCNN-like scale."""
        w_scale = float(target_wh[0]) / img_wh[0]
        h_scale = float(target_wh[1]) / img_wh[1]
        if scale_type == "W":
            return w_scale
        elif scale_type == "H":
            return h_scale
        elif scale_type == "MIN":
            return min(w_scale, h_scale)
        elif scale_type == "MAX":
            return max(w_scale, h_scale)
        else:
            raise ValueError("Unknow scale_type:{}".format(scale_type))

    W, H = src_wh

    if norm_scale is None:
        scale = get_scale((W, H), norm_wh, scale_type)
    else:
        scale = norm_scale

    roi_mat = rand_scale_affine_mat(
        src_wh=src_wh,
        dst_wh=target_wh,
        min_scale=scale * rand_scale_range[0],
        max_scale=scale * rand_scale_range[1],
        align_type="center" if center_aligned else "rand",
    )
    src_roi = np.array([(0, 0), src_wh])
    point_affine_ts = Point2DAffineTransform()
    target_roi = point_affine_ts(src_roi, roi_mat[0:2])
    return get_affine_by_roi(
        src_wh,
        target_wh,
        src_roi,
        target_roi,
        center_aligned=center_aligned,
        **kwargs
    )
