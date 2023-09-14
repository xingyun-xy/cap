import os

from cap.registry import OBJECT_REGISTRY
from cap.visualize.bbox3d import draw_bbox3d

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
import json
import random

import cv2
import numpy as np
from capbc.utils import _as_list

from cap.core.anno_ts_utils import IgnoreRegion, ImageRecord, RecordInstance

__all__ = [
    "VizDenseBoxDetAnno",
    "VizRoiDenseBoxDetAnno",
    "plot_densebox_image_record_bbox",
]  # noqa


@OBJECT_REGISTRY.register
class VizDenseBoxDetAnno(object):
    def __init__(
            self,
            viz_class_id,
            class_name,
            save_flag=False,
            save_path=None,
            lt_point_id=0,
            rb_point_id=2,
            **kwargs
    ):  # noqa
        self.save_flag = save_flag
        self.save_path = save_path
        self.viz_class_id = viz_class_id
        self.class_name = class_name
        self.lt_point_id = lt_point_id
        self.rb_point_id = rb_point_id
        self._mkdir()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.vis_config = kwargs.get('vis_configs', None)

    def _mkdir(self):
        if self.save_flag and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, img, anno):
        if 'image_key' not in anno:  # 2D
            image_name = anno["img_url"].split("/")[-1]
            assert plt is not None, "cannot import `matplotlib`"
            plot_densebox_image_record_bbox(
                img,
                anno,
                lt_point_id=self.lt_point_id,
                rb_point_id=self.rb_point_id,
                viz_class_id=self.viz_class_id,
                class_names=self.class_name,
                viz_normal=True,
                viz_hard=True,
                viz_ignore=True,
            )
            # save image
            if self.save_flag:
                plt.savefig(os.path.join(self.save_path, image_name))
        # plt.show()
        else:  # 3D
            assert self.vis_config is not None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_name = anno['image_key']
            calib = np.array(anno['meta']['calib'])
            distCoeffs = np.array(anno['meta']['distCoeffs'])
            for i in anno['objects']:
                cls_name = self.class_name[i['category_id'] - 1]
                vis_config = self.vis_config[cls_name]
                x, y = i['bbox_2d'][:2]
                an3d = i['in_camera']
                yaw = an3d['rotation_y']
                dimension = np.array(an3d['dim'])
                location = np.array(an3d['location'])
                location[1] += dimension[0] / 2
                draw_bbox3d(img,
                            location,
                            dimension,
                            yaw,
                            vis_config["color"],
                            vis_config["thickness"],
                            calib,
                            distCoeffs)
                cv2.putText(img, cls_name, (int(x), int(y) + 10), self.font, 0.6, vis_config["color"],
                            vis_config["thickness"])
            cv2.imwrite(os.path.join(self.save_path, image_name), img)


@OBJECT_REGISTRY.register
class VizRoiDenseBoxDetAnno(object):
    """
    Visualize roi densebox detection annotation.

    Parameters
    ----------
    viz_class_id : list/tuple of int
        Visualized class ids.
    class_name : list/tuple of str
        Class names.
    save_flag : bool, optional
        Whether save rendered image, by default False
    save_path : str, optional
        Where to save image, by default None
    lt_point_id : int, optional
        Point id of left top, by default 0
    rb_point_id : int, optional
        Point id of right bottom, by default 2
    roi_lt_point_id : int, optional
        Roi point id of left top, by default 10
    roi_rb_point_id : int, optional
        Roi point id of right bottom, by default 12
    """

    def __init__(
            self,
            viz_class_id,
            class_name,
            save_flag=False,
            save_path=None,
            lt_point_id=0,
            rb_point_id=2,
            roi_lt_point_id=10,
            roi_rb_point_id=12,
    ):  # noqa
        self.save_flag = save_flag
        self.save_path = save_path
        self.viz_class_id = viz_class_id
        self.class_name = class_name
        self.lt_point_id = lt_point_id
        self.rb_point_id = rb_point_id
        self.roi_lt_point_id = roi_lt_point_id
        self.roi_rb_point_id = roi_rb_point_id
        self._mkdir()

    def _mkdir(self):
        if self.save_flag and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, img, anno):
        image_name = anno["img_url"].split("/")[-1]
        assert plt is not None, "cannot import `matplotlib`"
        ax = plot_densebox_image_record_bbox(
            img,
            anno,
            lt_point_id=self.lt_point_id,
            rb_point_id=self.rb_point_id,
            viz_class_id=self.viz_class_id,
            class_names=self.class_name,
            viz_normal=True,
            viz_hard=True,
            viz_ignore=True,
        )
        ax = plot_densebox_image_record_bbox(
            img,
            anno,
            lt_point_id=self.roi_lt_point_id,
            rb_point_id=self.roi_rb_point_id,
            viz_class_id=self.viz_class_id,
            class_names=self.class_name,
            viz_normal=True,
            viz_hard=True,
            viz_ignore=True,
            ax=ax,
        )
        # save image
        if self.save_flag:
            plt.savefig(os.path.join(self.save_path, image_name))
        # plt.show()


def plot_densebox_image_record_bbox(
        img,
        image_record,
        lt_point_id,
        rb_point_id,
        viz_class_id=None,
        class_names=None,
        viz_normal=True,
        colors=None,
        viz_hard=False,
        hard_color=(0, 255, 255),
        viz_ignore=False,
        ignore_color=(255, 255, 255),
        scale_x=1.0,
        scale_y=1.0,
        ax=None,
):
    """Plot bbox for densebox ImageRecord.

    Parameters
    ----------
    img: np.ndarray
        Image with shape (H, W, C)
    image_record: ImageRecord
        densebox annotation.
    lt_point_id: int
        Left top point index.
    rb_point_id: int
        Right bottom point index.
    viz_class_id: list/tuple of int, or None
        Visualized class id.

        .. note::

            Start from 1.
    class_names: list/tuple of str, or None
        Class names.
    viz_normal: bool
        Whether to visualize normal instance.
    colors: dict of list/tuple of int
        Colors for normal instance. Looks like

        .. code-block:: none

            {
                0: (0, 0, 255),
                1: (0, 255, 0),
                ...
            }
    viz_hard: bool
        Whether to visualize hard instance.
    hard_color: list/tuple of int
        Colors for hard instance.
    viz_ignore: bool
        Whether to visualize ignore regions.
    ignore_color: list/tuple of int
        Colors for ignore regions.
    scale_x: float
        Rescale factor in width.
    scale_y: float
        Rescale factor in height.
    ax: matplotlib axes, or None
        You can reuse previous axes if provided.

    Returns
    -------
    matplotlib axes
        The ploted axes.
    """

    viz_class_id = _as_list(viz_class_id) if viz_class_id is not None else None
    if viz_class_id:
        for i in viz_class_id:
            assert i > 0, "class id should begin from 1"
    class_names = _as_list(class_names) if class_names is not None else None
    if isinstance(viz_class_id, (list, tuple)) and isinstance(
            class_names, (list, tuple)
    ):
        assert len(viz_class_id) == len(class_names), "%d vs. %d" % (
            len(viz_class_id),
            len(class_names),
        )

    if isinstance(image_record, str):
        image_record = ImageRecord(init_dict=json.loads(image_record))
    elif isinstance(image_record, dict):
        image_record = ImageRecord(init_dict=image_record)
    else:
        assert isinstance(
            image_record, ImageRecord
        ), "expected ImageRecord, get %s" % str(type(image_record))

    normal_bboxes = []
    normal_class_ids = []
    hard_bboxes = []
    ignore_bboxes = []
    for _, instance in enumerate(image_record.instances):
        if isinstance(instance, dict):
            instance = RecordInstance(init_dict=instance)
        class_id = instance.class_id[0]
        if viz_class_id is not None and class_id not in viz_class_id:
            continue
        is_hard = instance.is_hard[0] == 1
        if not is_hard and not viz_normal:
            continue
        if is_hard and not viz_hard:
            continue
        # get point id
        if is_hard:
            hard_bboxes.append(
                [
                    instance.points_data[lt_point_id][0],
                    instance.points_data[lt_point_id][1],
                    instance.points_data[rb_point_id][0],
                    instance.points_data[rb_point_id][1],
                ]
            )
        else:
            normal_bboxes.append(
                [
                    instance.points_data[lt_point_id][0],
                    instance.points_data[lt_point_id][1],
                    instance.points_data[rb_point_id][0],
                    instance.points_data[rb_point_id][1],
                ]
            )
            normal_class_ids.append(class_id - 1)  # make class id begin from 0
    if viz_ignore:
        for ignore_region in image_record.ignore_regions:
            if isinstance(ignore_region, dict):
                ignore_region = IgnoreRegion(init_dict=ignore_region)
            ignore_bboxes.append(
                [
                    ignore_region.left_top[0],
                    ignore_region.left_top[1],
                    ignore_region.right_bottom[0],
                    ignore_region.right_bottom[1],
                ]
            )
    normal_bboxes = np.array(normal_bboxes)
    normal_class_ids = np.array(normal_class_ids)
    hard_bboxes = np.array(hard_bboxes)
    ignore_bboxes = np.array(ignore_bboxes)

    # rescale image and bbox
    if scale_x != 1.0 or scale_y != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
        normal_bboxes = _rescale_bbox(normal_bboxes, scale_x, scale_y)
        hard_bboxes = _rescale_bbox(hard_bboxes, scale_x, scale_y)
        ignore_bboxes = _rescale_bbox(ignore_bboxes, scale_x, scale_y)

    if viz_normal:
        colors = _normalize_color(colors)
        ax = gcv_plot_bbox(
            img,
            normal_bboxes,
            labels=normal_class_ids,
            class_names=class_names,
            colors=colors,
            ax=ax,
        )
    if viz_hard:
        hard_color = _normalize_color(hard_color)
        ax = gcv_plot_bbox(
            img,
            hard_bboxes,
            colors={0: hard_color},
            labels=np.zeros((hard_bboxes.shape[0])),
            ax=ax,
            class_names=["hard"],
        )
    if viz_ignore:
        ignore_color = _normalize_color(ignore_color)
        ax = gcv_plot_bbox(
            img,
            ignore_bboxes,
            colors={0: ignore_color},
            labels=np.zeros((ignore_bboxes.shape[0])),
            ax=ax,
            class_names=["ignore"],
        )
    return ax


def _rescale_bbox(bbox, scale_x, scale_y):
    """Rescale bbox."""
    if bbox.size > 0:
        bbox[:, 0] *= scale_x
        bbox[:, 1] *= scale_y
        bbox[:, 2] *= scale_x
        bbox[:, 3] *= scale_y
    return bbox


def _normalize_color(colors):
    """Normalize rgb color to within [0.0, 1.0]."""
    if colors is None:
        return colors
    if isinstance(colors, dict):
        for key in colors:
            colors[key] = _normalize_color(colors[key])
    elif isinstance(colors, (list, tuple)):
        colors = [i / 255.0 for i in colors]
    else:
        raise NotImplementedError(
            "unsupported type %s for color" % str(type(colors))
        )
    return colors


def gcv_plot_bbox(
        img,
        bboxes,
        scores=None,
        labels=None,
        thresh=0.5,
        class_names=None,
        colors=None,
        ax=None,
        reverse_rgb=False,
        absolute_coordinates=True,
):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided.
        Scores with less than `thresh` will be ignored in display,
        this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...},
        otherwise random colors will be substituted.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered,
        otherwise coordinates are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """
    from matplotlib import pyplot as plt

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError(
            "The length of labels and bboxes mismatch, {} vs {}".format(
                len(labels), len(bboxes)
            )
        )
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError(
            "The length of scores and bboxes mismatch, {} vs {}".format(
                len(scores), len(bboxes)
            )
        )

    ax = plot_image(img, ax=ax, reverse_rgb=reverse_rgb)

    if len(bboxes) < 1:
        return ax

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = {}
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap("hsv")(cls_id / len(class_names))
            else:
                colors[cls_id] = (
                    random.random(),
                    random.random(),
                    random.random(),
                )
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5,
        )
        ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ""
        score = "{:.3f}".format(scores.flat[i]) if scores is not None else ""
        if class_name or score:
            ax.text(
                xmin,
                ymin - 2,
                "{:s} {:s}".format(class_name, score),
                bbox={
                    "facecolor": colors[cls_id],
                    "alpha": 0.5,
                },
                fontsize=12,
                color="white",
            )
    return ax


def plot_image(img, ax=None, reverse_rgb=False):
    """Visualize image.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.

    Returns
    -------
    matplotlib axes
        The ploted axes.
    Examples
    --------
    from matplotlib import pyplot as plt
    ax = plot_image(img)
    plt.show()
    """
    from matplotlib import pyplot as plt

    if ax is None:
        # create new axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax


@OBJECT_REGISTRY.register
class VizKpsDetAnno(object):
    """
    Visualize key point detection annotation.

    Parameters
    ----------
    num_kps : int
        The number of key points
    task_name : str
        What kind of key point task, possible values are
        {'cyclist_2_kps', 'vehicle_2_kps', 'vehicle_12_kps',
        'vehicle_8_kps','vehicle_4_kps'}
    save_flag : bool, optional
        Whether save rendered image, by default False
    save_path : str, optional
        Where to save image, by default None
    use_cam_params : bool, optional
        Whether visualize with camera parameter.
    """

    def __init__(self, num_kps, task_name, save_flag=False, save_path=None):
        self.num_kps = num_kps
        self.task_name = task_name
        self.save_flag = save_flag
        self.save_path = save_path

        self._mkdir()

    def _mkdir(self):
        if self.save_flag and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, img, anno):
        gt_contents = anno
        image_abs_path = gt_contents["image"]
        image_name = image_abs_path.strip().split("/")[-1]

        # check object number
        num_obj = len(gt_contents["boxes"])
        if num_obj <= 0:
            return

        # traverse each object instance
        for i in range(int(num_obj)):
            gt_obj_cur = gt_contents["boxes"][i]
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = map(float, gt_obj_cur)
            keypoints = gt_contents["keypoints"]

            wheel_back = keypoints[i][0:2]
            wheel_front = keypoints[i][3:5]

            img = visualize_onraw_img_2_kps(
                image=img,
                bbox=[bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax],
                lms_wheel_back=wheel_back,
                lms_wheel_front=wheel_front,
                ind=i,
            )

        # save image
        if self.save_flag:
            cv2.imwrite(os.path.join(self.save_path, image_name), img)
        return img


def visualize_onraw_img_2_kps(
        image, bbox, lms_wheel_back, lms_wheel_front, ind, norm_error=0.0
):
    """Draw 2 anno kps on the raw image."""
    thick = 2
    xmin, ymin, xmax, ymax = bbox

    # draw wheel line
    cv2.line(
        image,
        (int(lms_wheel_back[0]), int(lms_wheel_back[1])),
        (int(lms_wheel_front[0]), int(lms_wheel_front[1])),
        (0, 255, 0),
        thick + 1,
    )

    # red
    cv2.circle(
        image, (int(lms_wheel_back[0]), int(lms_wheel_back[1])), 3, (0, 0, 255)
    )

    # blue
    cv2.circle(
        image,
        (int(lms_wheel_front[0]), int(lms_wheel_front[1])),
        3,
        (255, 0, 0),
    )

    # bbox
    cv2.rectangle(
        image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 140, 255), 2
    )

    # draw annotation
    cv2.putText(
        image,
        "id:" + str(ind),
        (int(xmin), int(ymin) - 20),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 140, 255),
    )
    cv2.putText(
        image,
        "lme:" + "%.2f" % norm_error,
        (int(xmin), int(ymin)),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (0, 140, 255),
    )
    cv2.putText(
        image,
        str(0),
        (
            max(int(lms_wheel_back[0]) - 20, 0),
            max(int(lms_wheel_back[1]) - 20, 0),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )
    cv2.putText(
        image,
        str(1),
        (
            max(int(lms_wheel_front[0]) - 20, 0),
            max(int(lms_wheel_front[1]) - 20, 0),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3,
    )

    return image
