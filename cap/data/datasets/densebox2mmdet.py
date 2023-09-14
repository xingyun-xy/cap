import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from cap.registry import build_from_registry
from cap.utils.apply_func import _as_list

try:
    from cap.datasets import CustomDataset

    BASEDATE = CustomDataset
    _MMDET_IMPORTED = True

except ImportError:
    BASEDATE = Dataset
    _MMDET_IMPORTED = False

from typing import List

__all__ = ["DenseboxDataset2MMDet"]


class DenseboxDataset2MMDet(BASEDATE):
    """CAP DenseboxDataset interface for mmdetection.

    Args:
        dataset (dict): cfg of CAP `DenseboxDataset`,
            Includes:
                data_path : Path of data relative to bucket path.
                anno_path : Path of annotation.
                transforms : List of transform.
                to_rgb: Convert bgr(cv2 imread) to rgb.
                task_type: Consist of 'detection', 'segmentation'
                class_id : the rec's class id, 1base
                category : the used category, 0base
                rec_idx_file_path: index file related to data_path.
                    Used only when there is already index file somewhere.
                disable_default_densebox_log: Disable default print output
                    from `LegacyDenseBoxImageRecordDataset`. Default is True.
        classes (tuple): Specify classes to load.
        pipeline (list[dict]): Processing pipeline of mmdetection.
        test_mode (bool): If set True, annotation will not be loaded.
        filter_empty_gt (bool): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`. Defaults to True.
    """

    def __init__(
        self,
        dataset: dict,
        classes: tuple = None,
        pipeline: List[dict] = None,
        test_mode: bool = False,
        filter_empty_gt: bool = True,
    ) -> None:

        if not _MMDET_IMPORTED:
            raise ModuleNotFoundError(" `mmcv` and `cap` are required.")

        assert dataset, "`dataset` can not be None."
        assert isinstance(
            dataset, (dict, list)
        ), f"`dataset` should be a dict or list, but get {type(dataset)}."  # noqa E501

        self.task_type = None

        dataset_cfgs = _as_list(dataset)
        datasets = []
        for cap_cfg in dataset_cfgs:
            assert isinstance(
                cap_cfg, dict
            ), f"element in `dataset` should be dict, but get {type(cap_cfg)}"  # noqa E501
            assert (
                cap_cfg.get("transform", None) is None
            ), f"`transform` should be None, but get {cap_cfg.get('transform')}"  # noqa E501
            task_type = cap_cfg.get("task_type", "detection")
            if self.task_type is None:
                self.task_type = task_type
            else:
                assert (
                    self.task_type == task_type
                ), "`task_type` in different datasets should be same"
            dataset = build_from_registry(cap_cfg)
            datasets.append(dataset)

        self.dataset = ConcatDataset(datasets)

        self.is_test_mode = test_mode
        self.flag = np.ones(len(self.dataset), dtype=np.uint8)
        self.indexs = list(range(len(self.dataset)))

        super(DenseboxDataset2MMDet, self).__init__(
            ann_file=None,
            pipeline=pipeline,
            data_root=None,
            classes=classes,
            img_prefix="",
            seg_prefix=None,
            proposal_file=None,
            test_mode=True,
            filter_empty_gt=filter_empty_gt,
            file_client_args={"backend": "disk"},
        )

    def load_annotations(self, ann_file):
        # overwrite load_annotations
        return None

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        index = idx
        if self.is_test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            self.indexs[index] = idx
            return data

    def _check_gt(self, ann):
        """Check if there is gt_bbox."""
        if self.task_type == "detection":
            gt = ann["bboxes"]
        else:
            # TODO(): segmentation
            gt = None
        if gt is None or not gt.any():
            return False

        return True

    def __len__(self):
        return len(self.dataset)

    def get_ann_info(self, idx):
        if not self.is_test_mode:
            # training
            self.__getitem__(idx)
            idx = self._get_real_idx(idx)
        return self._convert_format(self.dataset[idx])["ann"]

    def get_cat_ids(self, idx):
        if not self.is_test_mode:
            # training
            self.__getitem__(idx)
            idx = self._get_real_idx(idx)
        return (
            self._convert_format(self.dataset[idx])["ann"]["labels"]
            .astype(np.int)
            .tolist()
        )

    def _get_real_idx(self, idx):
        return self.indexs[idx]

    def _convert_format(self, data):
        """Convert data format from CAP to mmdetection."""

        ann = {}
        if self.task_type.lower() == "detection":
            ann["bboxes"] = data["gt_bboxes"]
            ann["labels"] = data["gt_classes"]

        infos = {}
        infos["img_id"] = data["img_id"]
        infos["filename"] = data["img_name"]
        infos["width"] = data["img_width"]
        infos["height"] = data["img_height"]
        infos["ann"] = ann

        return infos

    def prepare_train_img(self, idx):
        # overwrite prepare_train_img

        idx = self._get_real_idx(idx)

        img_info = self._convert_format(self.dataset[idx])
        ann_info = self.get_ann_info(idx)

        if self.filter_empty_gt and not self._check_gt(ann_info):
            return None

        results = {"img_info": img_info, "ann_info": ann_info}

        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        self.pre_pipeline(results)

        # MMDetection LoadImageFromFile
        results = self._convert_load_img_from_file(self.dataset[idx], results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        # overwrite prepare_train_img
        idx = self._get_real_idx(idx)
        img_info = self._convert_format(self.dataset[idx])
        results = {"img_info": img_info}
        if self.proposals is not None:
            results["proposals"] = self.proposals[idx]
        self.pre_pipeline(results)

        results = self._convert_load_img_from_file(self.dataset[idx], results)
        return self.pipeline(results)

    def _convert_load_img_from_file(self, data, results):
        """Convert data format as to `LoadImageFromFile` in mmdetection."""

        img = data["img"]

        results["filename"] = data["img_name"]
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = data["img_shape"]
        results["ori_shape"] = data["img_shape"]
        results["img_fields"] = ["img"]

        return results

    def evaluate(
        self,
        results,
        metric="mAP",
        logger=None,
        proposal_nums=(100, 300, 1000),
        iou_thr=0.5,
        scale_ranges=None,
    ):
        raise RuntimeError("`evaluate` is not implemented.")


try:
    from cap.datasets.builder import DATASETS

    DATASETS.register_module(
        module=DenseboxDataset2MMDet, name="DenseboxDataset2MMDet"
    )

except Exception as e:
    print(str(e))
