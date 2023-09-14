import os
import pickle

import cv2
import numpy as np
import pytest

from cap.evaluation import real3d


@pytest.mark.skipif(True, reason="not support yet")
def test_real3d_evaluation(tmpdir):
    dataset_fpath = ""
    prediction_fpath = "./eval_res/nuscenes_test/val/nuscenes_test/val/cap_eval/result.tar"
    setting_fpath = "projects/panorama/configs/eval/vehicle_heatmap_3d_detection.yaml"
    result = real3d.evaluate(dataset_fpath, prediction_fpath, setting_fpath,
                             tmpdir)
    assert isinstance(result, dict)


if __name__ == '__main__':
    test_real3d_evaluation("test_uni_real3d")
