from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from changan_plugin_pytorch.nn.functional import batched_nms

from cap.core.data_struct.app_struct import (
    DetObject,
    DetObjects,
    build_task_struct,
)
from cap.core.data_struct.base_struct import (
    ClsLabels,
    DetBoxes2D,
    DetBoxes3D,
    Lines2D,
    MultipleBoxes2D,
    Points2D_2,
)
from cap.registry import OBJECT_REGISTRY

CLASS_MAP = {
    "detection": DetBoxes2D,
    "classification": ClsLabels,
    "line": Lines2D,
    "detection_3d": DetBoxes3D,
    "kps": Points2D_2,
    "roi_detection": MultipleBoxes2D,
}


@OBJECT_REGISTRY.register
class RoIDecoder(nn.Module):
    """RoI Module Decoder.

    This is a multi-task decoder, collecting predictions from
    multiple outputs which provide various descriptions of
    aligned objects.
    """

    def __init__(
        self,
        task_descs: List[Dict[str, str]],
        input_padding: Sequence[int] = (0, 0, 0, 0),
        nms_threshold: Optional[float] = None,
        score_threshold: Optional[float] = None,
        transforms: Optional[List] = None,
    ):
        super().__init__()
        det_key = None
        for t_name, (_, t_type) in task_descs.items():
            if t_type == "detection":
                assert det_key is None, "1 detection only"
                det_key = t_name
        assert det_key is not None
        self.det_key = det_key
        self.task_descs = task_descs
        assert len(input_padding) == 4
        self.input_padding = input_padding
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        _, self.TASK_STRUCT = build_task_struct(
            "DetObject",
            "DetObjects",
            [
                (t_name, CLASS_MAP[t_type])
                for t_name, (_, t_type) in task_descs.items()
            ],
            bases=(DetObject, DetObjects),
        )
        self.transforms = transforms if transforms is not None else []

    def forward(self, *results: Dict[str, torch.Tensor]):
        assert len(results) == len(self.task_descs)
        pick_results = {
            task: res[desc[0]]
            for res, (task, desc) in zip(results, self.task_descs.items())
        }

        lengths = [len(res) for res in pick_results.values()]
        assert max(lengths) == min(lengths)

        batch_tmp = [{} for _ in range(max(lengths))]
        for t_name, res in pick_results.items():
            for i, r in enumerate(res):
                batch_tmp[i][t_name] = r

        # NMS on detection results in loop
        batch_results = []
        for res in batch_tmp:
            res_struct = self.TASK_STRUCT(**res)

            if self.nms_threshold is not None:
                if self.score_threshold is not None:
                    res_struct = res_struct.filter_by_lambda(
                        lambda x: getattr(x, self.det_key).scores
                        > self.score_threshold
                    )
                boxes: DetBoxes2D = getattr(res_struct, self.det_key)
                keep_idx = batched_nms(
                    boxes.boxes,
                    boxes.scores,
                    boxes.cls_idxs,
                    self.nms_threshold,
                )
                res_struct = res_struct[keep_idx]
                res_struct.inv_pad(*self.input_padding)

            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    res_struct = transform.inverse_transform(res_struct)

            batch_results.append(res_struct)

        return batch_results
