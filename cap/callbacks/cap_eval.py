# Copyright (c) Changan Auto. All rights reserved.

import json
import logging
import os
import pickle
import shutil
import tarfile
import uuid
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Optional

import cv2
import torch

from cap.registry import OBJECT_REGISTRY
from cap.evaluation import seg, real3d
from cap.utils.apply_func import _as_list
from cap.utils.distributed import get_dist_info, get_global_out, rank_zero_only
from .callbacks import CallbackMixin

__all__ = ["CAPEvalTaskType", "CAPEval"]

logger = logging.getLogger(__name__)


class CAPEvalTaskType(Enum):
    """CAPEval Task Type.
    """

    Det2D = "Detection 2D"
    SEG = "Semantic Segmentation"
    Real3D = "Real3D"
    Det3D = "Detection 3D"
    KPS = "Keypoints Detection Oks"
    LINE = "Keypoints_Ground_Line"
    BevSeg = "BevSeg"
    Bev3D = "Detection 3D"


@dataclass
class CAPEvalDataDesc:
    name: Optional[str] = None
    dataset_id: Optional[str] = None
    task_type: Optional[CAPEvalTaskType] = None

    def __post_init__(self):
        if self.dataset_id is not None:
            self.dataset_id = _as_list(self.dataset_id)


def _get_data_desc_save_key(data_desc):
    if data_desc.name is None:
        assert data_desc.dataset_id is not None
        return data_desc.dataset_id
    else:
        return data_desc.name


class CAPEvalHandler(ABC):
    """Basic CAP Eval Handler.

    Args:
        root: Root dir path.
        data_desc: Dataset description.
        prediction_name: Prediction name.
        prediction_tags: Prediction tags.
    """

    def __init__(
        self,
        root: str,
        input_yaml: str,
        data_path: str,
        data_desc: CAPEvalDataDesc,
        prediction_name: str,
        prediction_tags: str,
    ):

        self.root = root
        self.input_yaml = input_yaml
        self.data_path = data_path
        self.data_desc = data_desc

        self.prediction_name = prediction_name
        self.prediction_tags = prediction_tags
        self.result_file = None
        self.dir_name = None

        _, global_world_size = get_dist_info()
        self.global_world_size = global_world_size
        os.makedirs(self.root, exist_ok=True)

    def write(self, batch, model_outs):
        """Save predict result to file.

        Args:
            batch: The model's input.
            model_outs: The model's output.
        """
        raise NotImplementedError

    def check_file(self, epoch_id):
        """Rename file or establish file."""
        # self.result_file = f"{self.root}/result_{str(uuid.uuid4())}_{str(epoch_id)}.tar"  # noqa E501
        # self.dir_name = f"{self.root}/tmp_result_{str(uuid.uuid4())}_{str(epoch_id)}"  # noqa E501
        self.result_file = f"{self.root}/result.tar"  # noqa E501
        self.dir_name = f"{self.root}/tmp_result"  # noqa E501
        self.pkl_file = f"{self.dir_name}/result.pkl"
        os.makedirs(self.dir_name, exist_ok=True)

    def merge_file(self):
        """Merge result to json or tar img result and so on."""
        assert os.path.exists(self.dir_name)
        tar = tarfile.open(self.result_file, "w:gz")
        for file_name in os.listdir(self.dir_name):
            full_path = os.path.join(self.dir_name, file_name)
            tar.add(full_path)
        tar.close()
        shutil.rmtree(self.dir_name, ignore_errors=True)
        self.result_file = _as_list(self.result_file)


class CAPEvalDetHandler(CAPEvalHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()
        self.result_file = f"{self.root}/result.json"

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (xuefang.wang): support async
        if reformat_output_fn:
            rets = reformat_output_fn(batch, model_outs)
            assert isinstance(
                rets,
                List), f"`rets` should be List[Dict], but get {type(rets)}"
        else:
            # TODO(xuefang.wang): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself.")
        global_rank, global_out_rets = get_global_out(rets)
        if global_rank == 0:
            try:
                with open(self.result_file, "a") as fwrite:
                    for rets in global_out_rets:
                        for ret in rets:
                            assert isinstance(
                                ret, Dict
                            ), f"`ret` should be Dict, but get {type(ret)}"
                            image_key = ret.get("image_key", None)
                            assert image_key, "`image_key` can not be None"
                            if image_key in self._cache:
                                return
                            self._cache.add(image_key)
                            fwrite.write(json.dumps(ret) + "\n")
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.json_file}")

    def check_file(self, epoch_id):
        self.result_file = f"{self.root}/result_{str(uuid.uuid4())}_{str(epoch_id)}.json"  # noqa E501

    def merge_file(self):
        self.result_file = _as_list(self.result_file)


class CAPEvalSegHandler(CAPEvalHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()
        self.result_file = f"{self.root}/result.tar"
        self.dir_name = f"{self.root}/tmp_result"
        self.pkl_file = f"{self.dir_name}/result.pkl"

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (xuefang.wang): support async
        if reformat_output_fn:
            rets = reformat_output_fn(batch, model_outs)
            assert isinstance(
                rets,
                List), f"`rets` should be List[Dict], but get {type(rets)}"
        else:
            # TODO (xuefang.wang): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself.")

        for ret in rets:
            assert isinstance(
                ret, Dict), f"`ret` should be Dict, but get {type(ret)}"
            image_key = ret.get("image_name", None)
            assert image_key, "`image_name` can not be None"
            if image_key in self._cache:
                return
            self._cache.add(image_key)
            cv2.imwrite(
                os.path.join(self.dir_name, ret["image_name"]),
                ret["out_img"],
            )

        # rets.pop("predict_results")
        rets = pickle.dumps(rets)
        # rets = torch.save(rets)
        global_rank, global_out_rets = get_global_out(rets)
        if global_rank == 0:
            try:
                for ret in global_out_rets:
                    with open(self.pkl_file, "ab") as f:
                        f.write(ret)
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.pkl_file}")

        if self.global_world_size > 1:
            torch.distributed.barrier()

    def create_evaluation(self):
        """Evaluation."""
        assert self.data_desc.dataset_id, "`dataset_id` can not be None."
        assert self.result_file, "`result_file` can not be None."
        prediction_id = []
        dataset_fpath = self.data_path
        prediction_fpath = self.result_file[0]
        setting_fpath = self.input_yaml

        tmpdir = "tmp_eval_res"
        result = seg.evaluate(dataset_fpath, prediction_fpath, setting_fpath,
                              tmpdir)
        assert isinstance(result, dict)
        return prediction_id


class CAPEvalReal3DHandler(CAPEvalHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

        self.result_file = f"{self.root}/result.tar"
        self.dir_name = f"{self.root}/tmp_result"
        self.pkl_file = f"{self.dir_name}/result.pkl"

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (xuefang.wang): support async
        if reformat_output_fn:
            det_task_key = "vehicle_heatmap_3d_detection"
            ret = reformat_output_fn(batch,
                                     model_outs,
                                     det_task_key=det_task_key)
        else:
            # TODO (xuefang.wang): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself.")
        ret = pickle.dumps(ret)
        # ret = torch.save(ret, self.pkl_file)
        global_rank, global_out_rets = get_global_out(ret)
        if global_rank == 0:
            try:
                for ret in global_out_rets:
                    with open(self.pkl_file, "ab") as f:
                        f.write(ret)
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.pkl_file}")

    def create_evaluation(self):
        """Evaluation."""
        assert self.data_desc.dataset_id, "`dataset_id` can not be None."
        assert self.result_file, "`result_file` can not be None."
        prediction_id = []
        dataset_fpath = self.data_path
        prediction_fpath = self.result_file[0]
        setting_fpath = self.input_yaml

        tmpdir = "tmp_eval_res"
        result = real3d.evaluate(dataset_fpath, prediction_fpath,
                                 setting_fpath, tmpdir)
        assert isinstance(result, dict)
        return prediction_id


class CAPEvalBevSegHandler(CAPEvalHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

        self.result_file = f"{self.root}/result.tar"
        self.dir_name = f"{self.root}/tmp_result"
        self.pkl_file = f"{self.dir_name}/result.pkl"

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (xuefang.wang): support async
        if reformat_output_fn:
            ret = reformat_output_fn(batch, model_outs)
        else:
            # TODO(xuefang.wang): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself.")
        # save img
        img_name = ret["outputs"]["img_name"]
        for idx in range(len(img_name)):
            cv2.imwrite(
                os.path.join(self.dir_name, "bev_seg_" + img_name[idx]),
                ret["predict_results"][idx].astype("uint8"),
            )

        ret.pop("predict_results")
        ret = pickle.dumps(ret)
        global_rank, global_out_rets = get_global_out(ret)
        if global_rank == 0:
            try:
                for ret in global_out_rets:
                    with open(self.pkl_file, "ab") as f:
                        f.write(ret)
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.pkl_file}")


class CAPEvalBev3DHandler(CAPEvalHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

        self.result_file = f"{self.root}/result.tar"
        self.dir_name = f"{self.root}/tmp_result"
        self.pkl_file = f"{self.dir_name}/result.pkl"

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (xuefang.wang): support async
        if reformat_output_fn:
            ret = reformat_output_fn(batch, model_outs)
        else:
            # TODO(xuefang.wang): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself.")
        ret = pickle.dumps(ret)
        global_rank, global_out_rets = get_global_out(ret)
        if global_rank == 0:
            try:
                for ret in global_out_rets:
                    with open(self.pkl_file, "ab") as f:
                        f.write(ret)
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.pkl_file}")

    def _load_bev3d_pkl(self, pkl_path):
        pred_front = []
        pred_side = []
        f = open(pkl_path, "rb")
        while True:
            try:
                preds = pickle.load(f)
                pred_front.extend(preds["front_outputs"])
                pred_side.extend(preds["side_outputs"])
            except EOFError:
                break
        f.close()
        return pred_front, pred_side

    def merge_file(self):
        pred_front, pred_side = self._load_bev3d_pkl(self.pkl_file)

        front_pred = []
        front_dump_path = os.path.join(self.root, "pred_front.json")
        # Process front
        for front in pred_front:
            front_pred.append(json.dumps(front) + "\n")
        all_sample_num = len(front_pred)
        if all_sample_num > 0:
            with open(front_dump_path, "w") as w:
                for one in front_pred:
                    w.write(one)
        # Process side
        side_pred = []
        side_dump_path = os.path.join(self.root, "pred_side.json")
        for side in pred_side:
            side_pred.append(json.dumps(side) + "\n")
        all_sample_num = len(side_pred)
        if all_sample_num > 0:
            with open(side_dump_path, "w") as w:
                for one in side_pred:
                    w.write(one)

        self.result_file = [front_dump_path, side_dump_path]


TASK_TO_EVAL_TYPE = {
    'detection': CAPEvalTaskType.Det2D,
    'segmentation': CAPEvalTaskType.SEG,
    'detection_real_3d': CAPEvalTaskType.Real3D,
}

TASK_TO_HANDLER = {
    CAPEvalTaskType.Det2D: CAPEvalDetHandler,
    CAPEvalTaskType.SEG: CAPEvalSegHandler,
    CAPEvalTaskType.Real3D: CAPEvalReal3DHandler,
    CAPEvalTaskType.Det3D: CAPEvalDetHandler,
    CAPEvalTaskType.KPS: CAPEvalDetHandler,
    CAPEvalTaskType.LINE: CAPEvalDetHandler,
    CAPEvalTaskType.BevSeg: CAPEvalBevSegHandler,
    CAPEvalTaskType.Bev3D: CAPEvalBev3DHandler,
}

TASK_TO_HANDLER.update({k.value: v for k, v in TASK_TO_HANDLER.items()})


@OBJECT_REGISTRY.register
class CAPEval(CallbackMixin):
    """CAP Evaluation Callback.

    Args:
        output_root: Evaluation result output dir.
        prediction_name: Prediction name.
        prediction_tags: Prediction tags.
        cap_eval_dataset_name: Dataset name.
        cap_eval_dataset_id: Dataset id, `cap_eval_dataset_id` and
        `cap_eval_dataset_name` should have and only have one is not None.
        cap_eval_host: CAP Eval host.
        cap_eval_token: CAP Eval user token.
        reformat_output_fn: Callabel function to reformat model outpus.
        reformat_out_fn_kwargs: Custom params used in `reformat_output_fn`.
        reformat_input_fn: Callabel function to reformat input batch data.
        overwrite: Whether to overwrite existing prediction file.
    """

    def __init__(
        self,
        output_root: str,
        input_yaml: str,
        data_path: str,
        prediction_name: str,
        prediction_tags: Optional[List[str]] = None,
        cap_eval_dataset_name: Optional[List[str]] = None,
        cap_eval_dataset_id: Optional[List[int]] = None,
        cap_eval_type: Optional[str] = None,
        reformat_output_fn: Optional[Callable] = None,
        reformat_out_fn_kwargs: Optional[Dict] = None,
        reformat_input_fn: Optional[Callable] = None,
        with_epoch_id: bool = False,
        overwrite: Optional[bool] = True,
    ):

        super().__init__()
        self.output_root = output_root
        self.input_yaml = input_yaml
        self.data_path = data_path
        # cap_eval
        self.prediction_name = prediction_name
        self.prediction_tags = prediction_tags

        assert (
            cap_eval_dataset_id or cap_eval_dataset_name
        ), "`cap_eval_dataset_id` and `cap_eval_dataset_name` can not both be None."  # noqa E501
        assert not (cap_eval_dataset_id and cap_eval_dataset_name), (
            f"`cap_eval_dataset_id` and `cap_eval_dataset_name` should have"
            f"and only have one is not None, but get `cap_eval_dataset_id`: "
            f"{cap_eval_dataset_id}, `cap_eval_dataset_name`: {cap_eval_dataset_name}."  # noqa E501
        )

        self.cap_eval_dataset_id = (_as_list(cap_eval_dataset_id)
                                    if cap_eval_dataset_id else [])
        self.cap_eval_dataset_name = (_as_list(cap_eval_dataset_name)
                                      if cap_eval_dataset_name else [])

        self.reformat_outputs_fn = reformat_output_fn
        self.cap_eval_type = cap_eval_type
        self._handler = {}
        self._missing_datasets = set()
        self._prediction_ids = []
        self.overwrite = overwrite
        self.reformat_input_fn = reformat_input_fn
        self.reformat_out_fn_kwargs = reformat_out_fn_kwargs
        self.with_epoch_id = with_epoch_id

        self._init_data_desc()

    def _init_data_desc(self):
        task_type = None
        self.cap_eval_type
        task_type = TASK_TO_EVAL_TYPE.get(self.cap_eval_type, None)
        data_desc = CAPEvalDataDesc(
            name=self.cap_eval_dataset_id
            if not self.cap_eval_dataset_name else self.cap_eval_dataset_name,
            dataset_id=self.cap_eval_dataset_id,
            task_type=task_type,
        )
        if data_desc.dataset_id is None:
            raise ValueError("dataset id is required")
        self.data_desc = data_desc

    def _get_handler(self, data_desc: dict, epoch_id: int = None):

        key_list = _get_data_desc_save_key(data_desc)
        key = ",".join([str(key).strip() for key in key_list])
        if key not in self._handler:
            obj = TASK_TO_HANDLER.get(data_desc.task_type, None)
            if obj is None:
                raise ValueError(
                    f"Does not support cap eval task type {data_desc.task_type}"  # noqa
                )

            handler = obj(
                root="{}/{}/cap_eval".format(self.output_root, key),
                input_yaml=self.input_yaml,
                data_path=self.data_path,
                data_desc=data_desc,
                prediction_name=f"{self.prediction_name}_epoch{epoch_id}"
                if epoch_id is not None else self.prediction_name,
                prediction_tags=self.prediction_tags,
            )
            self._handler[key] = handler
        return self._handler[key]

    def on_epoch_begin(self, epoch_id, **kwargs):
        epoch_id = epoch_id if self.with_epoch_id else None
        handler = self._get_handler(self.data_desc, epoch_id=epoch_id)
        handler.check_file(epoch_id=epoch_id)

    def on_batch_begin(self, batch, **kwargs):
        if self.reformat_input_fn:
            self.reformat_input_fn(batch)

    def on_batch_end(self, batch, model_outs, **kwargs):

        if model_outs is None:
            self._missing_datasets.add(self.data_desc.dataset_id)
            return

        handler = self._get_handler(self.data_desc)
        handler.write(
            batch,
            model_outs,
            partial(self.reformat_outputs_fn, **self.reformat_out_fn_kwargs),
        )

    @rank_zero_only
    def on_epoch_end(self, **kwargs):
        for dataset_id in self.data_desc.dataset_id:
            if dataset_id in self._missing_datasets:
                return
        handler = self._get_handler(self.data_desc)
        handler.merge_file()
        prediction_id = handler.create_evaluation()
        logger.info(
            f"Create prediction {prediction_id} for dataset {self.data_desc.dataset_id}"  # noqa
        )
