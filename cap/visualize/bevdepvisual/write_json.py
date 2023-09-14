from cap.data.datasets.bevdepth import Quaternion
import tempfile
import os.path as osp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from cap.data.datasets.bevdepth import Box
import mmcv
import numpy as np
import os

def _format_bbox(results, img_metas, jsonfile_prefix=None):
    """Convert the results to the standard format.

    Args:
        results (list[dict]): Testing results of the dataset.
        jsonfile_prefix (str): The prefix of the output jsonfile.
            You can specify the output directory/filename by
            modifying the jsonfile_prefix. Default: None.

    Returns:
        str: Path of the output json file.
    """
    nusc_annos = {}
    mapped_class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian','traffic_cone',]

    print('Start to convert detection format...')

    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    
    for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
        boxes, scores, labels = det
        boxes = boxes
        sample_token = img_metas[sample_id]['token']
        trans = np.array(img_metas[sample_id]['ego2global_translation'])
        rot = Quaternion(img_metas[sample_id]['ego2global_rotation'])
        annos = list()
        for i, box in enumerate(boxes):
            name = mapped_class_names[labels[i]]
            center = box[:3]
            wlh = box[[4, 3, 5]]
            box_yaw = box[6]
            box_vel = box[7:].tolist()
            box_vel.append(0)
            quat = Quaternion(axis=[0, 0, 1], radians=box_yaw)
            nusc_box = Box(center, wlh, quat, velocity=box_vel)
            nusc_box.rotate(rot)
            nusc_box.translate(trans)
            if np.sqrt(nusc_box.velocity[0]**2 +
                        nusc_box.velocity[1]**2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck','trailer',]:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = DefaultAttribute[name]
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = DefaultAttribute[name]
            nusc_anno = dict(
                sample_token=sample_token,
                translation=nusc_box.center.tolist(),
                size=nusc_box.wlh.tolist(),
                rotation=nusc_box.orientation.elements.tolist(),
                velocity=nusc_box.velocity[:2],
                detection_name=name,
                detection_score=float(scores[i]),
                attribute_name=attr,
            )
            annos.append(nusc_anno)
        # other views results of the same frame should be concatenated
        if sample_token in nusc_annos:
            nusc_annos[sample_token].extend(annos)
        else:
            nusc_annos[sample_token] = annos
    nusc_submissions = {
        'meta': {'use_lidar': False, 'use_camera': True, 'use_radar': False, 'use_map': False, 'use_external': False},
        'results': nusc_annos,
    }
    mmcv.mkdir_or_exist(jsonfile_prefix)
    res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
    print('Results writes to', res_path)
    mmcv.dump(nusc_submissions, res_path)
    return res_path

def format_results( results,
                    img_metas,
                    output_dir = None, # json输出路径，自己指定
                    result_names=['img_bbox'], # 默认传值
                    jsonfile_prefix=None, # 默认传值
                    **kwargs):
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
    assert isinstance(results, list), 'results must be a list'

    if jsonfile_prefix is None:
        tmp_dir = tempfile.TemporaryDirectory()
        jsonfile_prefix = osp.join(tmp_dir.name, 'results')
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
        if '2d' in rasult_name:
            continue
        print(f'\nFormating bboxes of {rasult_name}')
        tmp_file_ = osp.join(jsonfile_prefix, rasult_name)
        if output_dir:
            result_files.update({rasult_name:_format_bbox(results, img_metas, output_dir)})
        else:
            result_files.update({rasult_name:_format_bbox(results, img_metas, tmp_file_)})
    return result_files, tmp_dir

if __name__ == "__main__":
    
    all_pred_results = list()
    all_img_metas = list()
    for validation_step_output in results: # 这个result就是保存的结果，可以考虑一个sample一个sample地处理 也就是这个result是一个sample的结果
        all_pred_results.append(validation_step_output[:3])
        all_img_metas.append(validation_step_output[3])
    
    os.makedirs("outputs",exist_ok = True)
    format_results(all_pred_results, all_img_metas, output_dir = "outputs") # 结果保存在outputs中