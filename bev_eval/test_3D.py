import sys

sys.path.append('/workspace/cap_develop/')

from evaluators.det_evaluators import DetNuscEvaluator
from utils.torch_dist import all_gather_object

import numpy as np
import json
from projects.panorama.configs.datasets.changan_lmdb_eval_datasets import bev_eval_datapath

data_root = bev_eval_datapath.data_root
anno_path = bev_eval_datapath.anno_path

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

# 整体探测范围：前，后，左右
DETECTION_RANGES = [50,50,50]

# 不同探测范围，选用不同d作为阈值
DETECTION_THRESH = {"1.0":[0,20],"2.0":[20,40],"3.0":[40,60],"4.0":[60,80]}

# 输出不同探测范围的结果
TEST_DIFF_RANGES = [[0,50],[0,20],[20,40],[40,50]]


def test_epoch_end(test_results, output_dir):

    evaluator = DetNuscEvaluator(class_names=CLASSES,
                                 eval_version='detection_ca_metric',
                                 data_root=data_root,
                                 anno_path=anno_path,
                                 plot_examples=0,
                                 d_iou = DETECTION_THRESH,
        						 ranges = DETECTION_RANGES,
								 test_ranges = TEST_DIFF_RANGES,
                                 output_dir=output_dir)

    test_step_outputs = np.load(test_results, allow_pickle=True).tolist()

    all_pred_results = list()
    all_img_metas = list()
    dataset_length = 0

    for test_step_output in test_step_outputs:
        dataset_length += len(test_step_output)  #one batch
        for i in range(len(test_step_output)):
            all_pred_results.append(test_step_output[i][0:3])
            all_img_metas.append(test_step_output[i][3])

    #dataset_length = len(self.val_dataloader().dataset)

    all_pred_results = sum(
        map(list, zip(*all_gather_object(all_pred_results))),
        [])[:dataset_length]
    all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                        [])[:dataset_length]

    evaluator.evaluate(all_pred_results, all_img_metas)


if __name__ == '__main__':
    #add scene label in tmp.npy file names
    # test_epoch_end("./bev_eval/tmp.npy","./results/")
    # test_epoch_end("./ret_list.npy","./results/")
    # test_epoch_end("./bev_eval/tmp_resave.npy","./results/")
    # test_epoch_end("./res_oneline.npy","./results/")
    test_epoch_end("../data/xfwang/bev_res_all.npy",
                   "./results/")
