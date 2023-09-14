import os
import pickle

import cv2
import numpy as np
import pytest

from cap.evaluation import seg


@pytest.mark.skipif(True, reason="not support yet")
def test_seg_evaluation(tmpdir):  #noqa
    dataset_fpath = (
        "/data/temp/eval/semantic_parsing/common_6000/camera_back_1000/")
    prediction_fpath = "./eval_res/semantic_parsing/common_6000/camera_back_1000/semantic_parsing/common_6000/camera_back_1000/cap_eval/result.tar"
    setting_fpath = "projects/panorama/configs/eval/default_segmentation.yaml"

    # dataset_fpath = "/data/temp/eval/lane_parsing/common_6000/camera_back_1000/"
    # prediction_fpath = "./eval_res/lane_parsing/common_6000/camera_back_1000/lane_parsing/common_6000/camera_back_1000/cap_eval/result.tar"
    # setting_fpath = "projects/panorama/configs/eval/lane_segmentation.yaml"
    result = seg.evaluate(dataset_fpath, prediction_fpath, setting_fpath,
                          tmpdir)
    assert isinstance(result, dict)


if __name__ == "__main__":
    test_seg_evaluation("test_uni_real3d")
