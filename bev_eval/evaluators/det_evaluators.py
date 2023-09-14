"""Modified from # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
"""
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

__all__ = ["DetNuscEvaluator"]


class DetNuscEvaluator:
    ErrNameMapping = {
        "x_err": "mAXE",
        "y_err": "mAYE",
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        # 'vel_err': 'mAVE',
        # 'attr_err': 'mAAE',
    }

    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    def __init__(
        self,
        class_names,
        eval_version="detection_ca_metric",
        data_root="./data/nuScenes",
        anno_path="./",
        plot_examples=0,
        d_iou={},
        ranges=[50, 50, 50],
        test_ranges=[[]],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        ),
        output_dir=None,
    ) -> None:
        self.eval_version = eval_version
        self.data_root = data_root
        self.anno_path = anno_path
        self.plot_examples = plot_examples
        if self.eval_version is not None:
            from .ca_metric.detection.config import config_factory

            self.eval_detection_configs = config_factory(self.eval_version)
        self.class_names = class_names
        self.modality = modality
        self.output_dir = output_dir
        self.d_iou = d_iou
        self.ranges = ranges
        self.test_ranges = test_ranges

    def _evaluate_single(
        self, result_path, logger=None, metric="bbox", result_name="pts_bbox"
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes

        from .ca_metric.detection.evaluate import CAEval

        output_dir = osp.join(*osp.split(result_path)[:-1])

        ca_eval = CAEval(
            data_root=self.data_root,
            anno_path=self.anno_path,
            ranges=self.ranges,
            config=self.eval_detection_configs,
            result_path=result_path,
            output_dir=output_dir,
            verbose=False,
        )

        # add different ranges and dist_thresh

        for range_d in self.test_ranges:

            ca_eval.main(
                range_d,
                self.d_iou,
                plot_examples=self.plot_examples,
                render_curves=False,
            )

    def format_results(
        self,
        results,
        img_metas,
        result_names=["img_bbox"],
        jsonfile_prefix=None,
        **kwargs,
    ):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
        result_files = dict()
        # refactor this.
        for rasult_name in result_names:
            # not evaluate 2D predictions on nuScenes
            if "2d" in rasult_name:
                continue
            print(f"\nFormating bboxes of {rasult_name}")
            tmp_file_ = osp.join(jsonfile_prefix, rasult_name)
            if self.output_dir:
                result_files.update(
                    {
                        rasult_name: self._format_bbox(
                            results, img_metas, self.output_dir
                        )
                    }
                )
            else:
                result_files.update(
                    {rasult_name: self._format_bbox(results, img_metas, tmp_file_)}
                )
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        img_metas,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(
            results, img_metas, result_names, jsonfile_prefix
        )
        if isinstance(result_files, dict):
            for name in result_names:
                print("Evaluating bboxes of {}".format(name))
                self._evaluate_single(result_files[name])
        elif isinstance(result_files, str):
            self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.

        update: 模型输出在ego下，评测也在ego下，不需要转到global
        模型直接输出size为 dydxdz, 评测时evalbox为dxdydz
        """
        ca_annos = {}
        mapped_class_names = self.class_names

        print("Start to convert detection format...")

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            sample_token = img_metas[sample_id]["token"]
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]  # dydxdz ---> dxdydz
                # wlh = box[[3, 4, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                ca_box = Box(center, wlh, quat, velocity=box_vel)
                if np.sqrt(ca_box.velocity[0] ** 2 + ca_box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self.DefaultAttribute[name]
                ca_anno = dict(
                    sample_token=sample_token,
                    translation=ca_box.center.tolist(),
                    size=ca_box.wlh.tolist(),
                    rotation=ca_box.orientation.elements.tolist(),
                    velocity=ca_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(ca_anno)
            # other views results of the same frame should be concatenated
            if sample_token in ca_annos:
                ca_annos[sample_token].extend(annos)
            else:
                ca_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": ca_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path
