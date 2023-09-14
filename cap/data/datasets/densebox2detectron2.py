import copy

import numpy as np
import torch

from cap.registry import build_from_registry
from cap.utils.apply_func import _as_list
from cap.utils.config import Config

try:
    from detectron2.data import (
        DatasetFromList,
        DatasetMapper,
        build_detection_train_loader,
    )
    from detectron2.data import detection_utils as utils
    from detectron2.data import transforms as T
    from detectron2.data.build import build_detection_test_loader
    from detectron2.structures import BoxMode

    BASEMAPPER = DatasetMapper
    _DETECTRON2_IMPORTED = True

except ImportError:
    BASEMAPPER = object
    _DETECTRON2_IMPORTED = False


__all__ = ["DatasetMapperToDetectron2", "build_detectron2_data_loader"]


class DatasetMapperToDetectron2(BASEMAPPER):
    """Custom DatasetMapper that can be used for detectron2.

    Args:
        cfg (CfgNode): CfgNode config of detectron2.
        is_train (bool): Is training or validation.
        filter_empty_annotations (bool): [description].
    """

    def __init__(
        self,
        cfg,
        is_train: bool = True,
        filter_empty_annotations: bool = True,
    ):

        if not _DETECTRON2_IMPORTED:
            raise ModuleNotFoundError("Not found module named `detection2`.")

        self.filter_empty_annotations = filter_empty_annotations

        image_format = cfg.INPUT.FORMAT
        # augmentations
        augmentations = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augmentations.insert(
                0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            )
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        use_instance_mask = cfg.MODEL.MASK_ON
        instance_mask_format = cfg.INPUT.MASK_FORMAT
        use_keypoint = cfg.MODEL.KEYPOINT_ON

        # Note: Do not support `create_keypoint_hflip_indices`
        # and cfg.MODEL.KEYPOINT_ON should be False for densebox dataset
        if use_keypoint:
            # utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
            raise ValueError("`cfg.MODEL.KEYPOINT_ON` should be False.")

        if cfg.MODEL.LOAD_PROPOSALS:
            precomputed_proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        else:
            precomputed_proposal_topk = None

        super().__init__(
            is_train=is_train,
            augmentations=augmentations,
            image_format=image_format,
            use_instance_mask=use_instance_mask,
            use_keypoint=use_keypoint,
            instance_mask_format=instance_mask_format,
            keypoint_hflip_indices=None,
            precomputed_proposal_topk=precomputed_proposal_topk,
            recompute_boxes=recompute_boxes,
        )

    def __call__(self, dataset_dict):
        """Process a batch of data.

        Args:
            dataset_dict (dict): Metadata of one image.

        Returns:
            dict: a format that builtin models in detectron2 accept.
        """

        assert self.image_format in ["BGR", "RGB"], (
            f"image format should be one of ['BGR', 'RGB'], "
            f"but get {self.image_format}."
        )

        dataset_dict = self._convert_format(dataset_dict)

        assert self.image_format == dataset_dict["image_format"], (
            f"`image format` in config is {self.image_format}, "
            f"which is different from real image format {dataset_dict['image_format']}"  # noqa E501
        )

        if dataset_dict is None:
            return None

        image = dataset_dict["img"]
        utils.check_image_size(dataset_dict, image)

        # TODO (): segmentation
        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to
        # shared-memory, but not efficient on large generic data
        # structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict

    def _convert_format(self, data):
        record = {}
        record["file_name"] = data["img_name"]
        record["image_id"] = data["img_id"]
        record["width"] = data["img_width"]
        record["height"] = data["img_height"]
        record["img"] = data["img"]
        record["image_format"] = data["color_space"]

        objs = []

        gt_boxes = data["gt_bboxes"]
        gt_labels = data["gt_classes"]

        if gt_boxes is not None and gt_boxes.any():
            for label, box in zip(gt_labels, gt_boxes):
                obj = {
                    "bbox": box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": label,
                }
                objs.append(obj)
        else:
            if self.filter_empty_annotations:
                return None

        record["annotations"] = objs

        return record


def from_yaml_file(dataset_cfg_path: str, task_name: str):

    cfg = Config.fromfile(dataset_cfg_path)
    assert (
        task_name in cfg
    ), f"task name: {task_name} not in dataset:{cfg.keys()}"

    datasets_cfg = cfg[task_name]

    return _as_list(datasets_cfg)


def _filter_dataset_cfg(cfg_dict, mode):

    cfg_dict_copy = copy.deepcopy(cfg_dict)
    path_cfg = cfg_dict_copy[mode]
    common_cfg = cfg_dict_copy["common"]
    final_cfg = dict(**path_cfg, **common_cfg)

    return final_cfg


def build_dataset(data_cfg, mode="train"):
    # for data_cfg in data_cfg_dict:
    data_cfg_path = data_cfg["rec_dataset_path_file"]
    task_name = data_cfg["task_name"]

    dataset_cfgs = from_yaml_file(data_cfg_path, task_name)

    datasets = []
    for dataset_cfg in dataset_cfgs:
        dataset_cfg = _filter_dataset_cfg(dataset_cfg, mode)
        dataset_cfg["type"] = "DenseboxDataset"
        dataset = build_from_registry(dataset_cfg)
        datasets.append(dataset)

    if len(datasets) > 1:
        dataset = DatasetFromList(datasets, copy=False)
    else:
        dataset = datasets[0]

    return dataset


def build_detectron2_data_loader(cfg, dataset_name=None, mode="train"):
    assert mode in [
        "train",
        "test",
    ], f"`mode` should be one of ['train', 'test'], but get {mode}"

    is_training = True if mode == "train" else False
    if mode == "train":
        data_cfg_dict = cfg.DATASETS.TRAIN
    else:
        data_cfg_dict = cfg.DATASETS.TEST

    assert data_cfg_dict, f"`DATASETS.{mode.upper()}` can not be None"
    assert (
        len(data_cfg_dict) == 1
    ), f"`DATASETS.{mode.upper()}` should have one dict value"

    if isinstance(data_cfg_dict[0], dict):

        # build cap dataset
        dataset = build_dataset(data_cfg_dict[0], mode=mode)

        # build mapper
        filter_empty_annotations = (
            cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS if is_training else False
        )

        mapper = DatasetMapperToDetectron2(
            cfg, is_training, filter_empty_annotations=filter_empty_annotations
        )

        # build dataloader
        if is_training:
            return build_detection_train_loader(
                cfg, dataset=dataset, mapper=mapper
            )
        else:
            return build_detection_test_loader(
                dataset=dataset,
                mapper=mapper,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
            )
    else:
        # use default build_detection_train_loader(cfg)
        if is_training:
            return build_detection_train_loader(cfg)
        else:
            return build_detection_test_loader(cfg, dataset_name)
