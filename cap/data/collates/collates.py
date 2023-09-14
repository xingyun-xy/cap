"""Copyright (c) Changan Auto. All rights reserved."""
import collections
import logging
import os.path as osp
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

try:
    from wenet.utils.common import IGNORE_ID
except ImportError:
    IGNORE_ID = -1

from cap.data.metas import img_metas
from cap.registry import OBJECT_REGISTRY

__all__ = [
    "collate_2d",
    "collate_3d",
    "collate_psd",
    "CocktailCollate",
    "collate_lidar",
    "collate_fn_bevdepth",
    "collate_fn_bevdepth_cooperate_pilot",
    "collate_fn_bevdepth_onnx",
]


def collate_psd(batch: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in parking slot detection(psd) task. \
        For collating data with inconsistent shapes.

    Args:
        batch (list): list of data.
    """
    elem = batch[0]
    # these key-value will skip default_collate
    list_key = ["img", "ori_img", "img_name", "label", "layout", "color_space"]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, collections.Mapping):
        return_data = {}
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch]
                if key == "img":
                    collate_data = torch.stack(collate_data, dim=0)
            else:
                collate_data = default_collate([d[key] for d in batch])

            return_data.update({key: collate_data})
        return return_data


def collate_2d(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in 2d task, for collating data with inconsistent shapes.

    Args:
        batch (list): list of data.
    """
    elem = batch[0]
    # these key-value will skip default_collate
    list_key = [
        "ig_regions",
        "gt_boxes",
        "parent_gt_boxes",
        "parent_ig_regions",
        "gt_flanks",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        unexpected_keys = []
        for key in elem:
            if key in list_key:
                collate_data = [torch.from_numpy(d[key]) for d in batch]
            else:
                collate_data = default_collate([d[key] for d in batch])
            if key not in img_metas:
                unexpected_keys.append(key)

            return_data.update({key: collate_data})
        logging.warning(
            f"{unexpected_keys} appear in keys of dataset."
            f"Please check whether it is an image task and meets expectations."
        )
        return return_data


def collate_3d(batch_data: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in bev task. \
    * If output tensor from dataset shape is (n,c,h,w),concat on \
        aixs 0 directly. \
    * If output tensor from dataset shape is (c,h,w),expand_dim on \
        axis 0 and concat.

    Args:
        batch (list): list of data.
    """
    if isinstance(batch_data[0], dict):
        result = {}
        for key in batch_data[0].keys():
            result[key] = collate_3d([d[key] for d in batch_data])
        return result
    elif isinstance(batch_data[0], (list, tuple)):
        return [collate_3d(data) for data in zip(*batch_data)]
    elif isinstance(batch_data[0], torch.Tensor):
        if len(batch_data[0].shape) == 4:
            return torch.cat(batch_data, dim=0)
        else:
            batch_data = [torch.unsqueeze(d, dim=0) for d in batch_data]
            return torch.cat(batch_data, dim=0)
    elif isinstance(batch_data[0], (str, int, float)):
        return batch_data
    else:
        raise TypeError


def collate_lidar(batch_list: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in rad task, for collating data with inconsistent shapes.
    Rad(Realtime and Accurate 3D Object Detection).

    First converts List[Dict[str, ...] or List[Dict]] to
    Dict[str, List], then process values whoses keys are
    related to training.

    Args:
        batch (list): list of data.
    """
    example_merged = collections.defaultdict(list)

    # 将batch_list中每个样本中相同的键值元素放置到一个list中.
    for example in batch_list:
        if isinstance(example, list):
            for subexample in example:
                for k, v in subexample.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)

    # 按照Key的不同，重新编排整理.
    # 每个key的elems功能不同，处理方式存在差异.
    batch_size = len(example_merged["metadata"])
    ret = {}
    for key, elems in example_merged.items():
        # 下述key，简单拼接并丈量化.
        if key in [
                "voxels",
                "num_points",
                "num_gt",
                "voxel_labels",
                "num_voxels",
                "points_num",
                "voxels_pillars",
                "num_points_pillars",
                "num_voxels_pillars",
                "pose",
                "whole_equation",
        ]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))

        # 下述key,将每个batch的所有elem拼接并张量化.
        # elems:List[List[array,...],...] -> List[tensor,tensor,...]
        elif key in [
                "sweep_voxels",
                "sweep_num_points",
                "sweep_num_voxels",
                "sweep_voxels_pillars",
                "sweep_num_points_pillars",
                "sweep_num_voxels_pillars",
        ]:
            batch_collated_list = []
            # idx为batch的索引.
            for idx in range(len(elems[0])):
                # 每个batch所有的elem放入到一个list中.
                batch_elem = [elem[idx] for elem in elems]
                batch_collated_list.append(
                    torch.tensor(np.concatenate(batch_elem, axis=0)))
            ret[key] = batch_collated_list

        # 下述key,将每个batch的所有elem拼接并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif key == "gt_boxes":
            task_max_gts = []
            # gt boxes为多个任务的监督列表.
            # task_id 为每个任务的gt_boxes.
            # 找到每个任务batch中最多gt_box的数量.
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            # 构建每个任务的监督array.
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    len_elem = len(elems[i][idx])
                    # 对每个gt_box赋索引值.
                    batch_task_gt_boxes3d[i, :len_elem, :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res

        elif key in ["metadata", "parsing"]:
            ret[key] = elems

        # 下述key,将每个elem在放回到所属batch的list中拼接并张量化.
        # elems::List[List[array,...],...] -> dict{str:tensor}
        elif key == "calib":
            ret[key] = {}
            # 将每个elem在放回到所属batch的list中.
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            # 拼接并张量化
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))

        # 下述key,每个elem都进行pad后拼接并张量化.
        # elems::List[List[array,...],...] -> Tensor
        elif key in [
                "coordinates",
                "points",
                "sample_points",
                "coordinates_pillars",
        ]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                  mode="constant",
                                  constant_values=i)
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))

        # 下述key,将每个batch的每个elem pad后拼接并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif key in [
                "sweep_coordinates",
                "sweep_points",
                "sweep_coordinates_pillars",
        ]:
            batch_collated_list = []
            for idx in range(len(elems[0])):
                batch_elem = [elem[idx] for elem in elems]
                coors = []
                for i, coor in enumerate(batch_elem):
                    coor_pad = np.pad(
                        coor,
                        ((0, 0), (1, 0)),
                        mode="constant",
                        constant_values=i,
                    )
                    coors.append(coor_pad)

                batch_collated_list.append(
                    torch.tensor(np.concatenate(coors, axis=0)))
            ret[key] = batch_collated_list

        # 下述key,将每个elem在放回到所属batch的list中拼接并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif (key in [
                "reg_targets",
                "reg_weights",
                "labels",
                "hm",
                "anno_box",
                "ind",
                "mask",
                "cap",
                "seg_hm",
                "kps_hm",
                "gt_boxes_tasks",
                "seg_loss_mask",
        ] or "hm_d" in key):
            ret[key] = collections.defaultdict(list)
            res = []
            for elem in elems:
                # 将每个元素放入到所属batch的list中.
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele))
            # 将每个batch的elem list进行stack后放入key的list.
            for _, vv in ret[key].items():
                res.append(torch.stack(vv))
            ret[key] = res

        # 下述key,每个elem都stack并张量化.
        # elems::List[List[array,...],...] -> List[tensor,tensor,...]
        elif key in ["gt_boxes_and_cls", "feature_trans"]:
            ret[key] = torch.tensor(np.stack(elems, axis=0))
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


@OBJECT_REGISTRY.register
class CocktailCollate(object):
    """CocktailCollate.

    鸡尾酒（多模）算法批量数据collate的Callable类.
    默认需要处理的是 dict 类型数据的列表。

    首先，将List[Dict[str, ...]]转换成Dict[str, List]
    然后，对dict中的 'images', 'audio', 'label' 跟训练相关的数据。
    进行 pad_sequence 操作。对 'tokens' 直接跳过。
    其他的key使用default_collate


    Args:
        ignore_id: 被忽略的标签ID, 默认使用wenet中的IGNORE_ID即-1.
                   处理标签数据时，使用IGNORE_ID的值作为padding值
        batch_first: 处理批量数据时, batch 的维度是否在第1位(数组编号0).
                     如果batch_first是True, 数组为 BxTx*
                     如果batch_first是False, 数组为 TxBx*
    """

    def __init__(self, ignore_id: int = IGNORE_ID, batch_first: bool = True):
        self.ignore_id = ignore_id
        self.batch_first = batch_first

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        # fetch keys
        keys_list = [tuple(sorted(elem.keys())) for elem in batch]
        keys = set(keys_list)
        assert len(keys) == 1, f"keys in should be same, but get {keys_list}"
        keys = keys.pop()
        # rearrange by key
        batch = {key: [elem[key] for elem in batch] for key in keys}
        for key, batch_ in batch.items():
            logging.debug(f"{key}, {type(batch_)}")
            if key in ["images", "audio"]:
                batch_ = pad_sequence(
                    batch_,
                    batch_first=self.batch_first,
                    padding_value=0,
                )
            elif key == "label":
                batch_ = pad_sequence(
                    batch_,
                    batch_first=self.batch_first,
                    padding_value=self.ignore_id,
                )
            elif key in ["tokens", "text"]:
                pass
            else:
                batch_ = default_collate(batch_)
            batch[key] = batch_
        return batch


def collate_fn_bevdepth(data, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()

    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        if is_return_depth:
            gt_depth = iter_data[10]
            depth_labels_batch.append(gt_depth)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
    ret_list = [
        torch.stack(imgs_batch),
        torch.stack(sensor2ego_mats_batch),
        torch.stack(intrin_mats_batch),
        torch.stack(ida_mats_batch),
        torch.stack(sensor2sensor_mats_batch),
        torch.stack(bda_mat_batch),
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    # return ret_list

    result = dict()  # noqa
    b, f, n, c, h, w = ret_list[0].shape
    if f == 1:
        ret_list[0] = ret_list[0].view(b * n, c, h, w)
    result["img"] = ret_list[0]
    result["sensor2ego_mats"] = ret_list[1]
    result["intrin_mats"] = ret_list[2]
    result["ida_mats"] = ret_list[3]
    result["sensor2sensor_mats"] = ret_list[4]
    result["bda_mat"] = ret_list[5]
    result["timestamps_batch"] = ret_list[6]
    result["img_metas_batch"] = ret_list[7]
    result["gt_boxes_batch"] = ret_list[8]
    result["gt_labels_batch"] = ret_list[9]
    if is_return_depth:
        b, n, c, h, w = ret_list[10].shape
        if b == 1:
            ret_list[10] = ret_list[10].view(n, c, h, w)
        result["depth_labels_batch"] = ret_list[10]
    return result


def collate_fn_bevdepth_onnx(data, is_return_depth=False):
    """FOR CURRENT ONNX CONVERSION ADDED BY ZWJ"""
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()

    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        if is_return_depth:
            gt_depth = iter_data[10]
            depth_labels_batch.append(gt_depth)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
    ret_list = [
        torch.stack(imgs_batch),
        torch.stack(sensor2ego_mats_batch),
        torch.stack(intrin_mats_batch),
        torch.stack(ida_mats_batch),
        torch.stack(sensor2sensor_mats_batch),
        torch.stack(bda_mat_batch),
        torch.stack(timestamps_batch),
        img_metas_batch,
        gt_boxes_batch,
        gt_labels_batch,
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    # return ret_list

    result = dict()  # noqa
    b, f, n, c, h, w = ret_list[0].shape
    if f == 1:
        ret_list[0] = ret_list[0].view(b * n, c, h, w)
    result["img"] = ret_list[0]
    result["sensor2ego_mats"] = ret_list[1]
    result["intrin_mats"] = ret_list[2]
    result["ida_mats"] = ret_list[3]
    result["sensor2sensor_mats"] = ret_list[4]
    result["bda_mat"] = ret_list[5]
    result["timestamps_batch"] = ret_list[6]
    result["img_metas_batch"] = ret_list[7]
    result["gt_boxes_batch"] = ret_list[8]
    result["gt_labels_batch"] = ret_list[9]
    import numpy as np
    """Required txt files loading part, DO NOT CHANGE A SIGNGLE ALPHABET! ADDED BY ZWJ """
    mlp_input = np.loadtxt("bev_input/mlp_input.txt")
    mlp_input = mlp_input.reshape((1, 1, 6, 27)).astype(np.float32)
    mlp_input = torch.from_numpy(mlp_input)

    circle_map = np.loadtxt("bev_input/circle_map.txt")
    circle_map = circle_map.reshape((1, 112, 16384)).astype(np.float32)
    circle_map = torch.from_numpy(circle_map)

    ray_map = np.loadtxt("bev_input/ray_map.txt")
    ray_map = ray_map.reshape((1, 216, 16384)).astype(np.float32)
    ray_map = torch.from_numpy(ray_map)

    result["mlp_input"] = mlp_input
    result["circle_map"] = circle_map
    result["ray_map"] = ray_map
    return result


def collate_fn_bevdepth_cooperate_pilot(data, is_return_depth=False):
    res = collate_fn_bevdepth(data, is_return_depth)
    # CAP orginal tasks needs img_shape as (n,c,h,w)
    img_shape = res["img"].shape
    print (img_shape)
    b = (img_shape[0] if res["img"].ndim == 4 else img_shape[0] *
         img_shape[1] * img_shape[2])
    res["calib"] = (res["intrin_mats"].view(b, 4, 4)[:,
                                                     0:3, :].to(torch.float32))
    # TODO distCoeffs temporary given，will be modified later   add by zmj
    res['distCoeffs'] = torch.zeros(b, 8, dtype=torch.float32)
    print (data[0][-1])
    res['ori_img_shape'] = data[0][-1].view(
        b, 3) if len(data) == 1 else torch.concat(
            ([i[-1] for i in data]), dim=0).view(b, 3)
    # ori_img=resized image，for later visualization,shape=(n,h,w,c)
    res['ori_img'] = res['img'].permute(0, 2, 3, 1)
    res["img_name"] = [
        osp.basename(i) for batch in res['img_metas_batch']
        for i in batch['file_name']
    ]

    res["img_height"] = torch.tensor([img_shape[2]] * b, dtype=torch.int64)
    res["img_width"] = torch.tensor([img_shape[3]] * b, dtype=torch.int64)
    # temporary given
    res["img_id"] = torch.tensor([0] * b, dtype=torch.int8)

    res["color_space"] = ["bgr"] * b
    res["layout"] = ["hwc"] * b
    res["img_shape"] = res["ori_img_shape"]
    res["pad_shape"] = res["ori_img_shape"]

    return res


def collate_fn_changanbev(data, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    bda_mats_batch = list()
    sensor2sensor_mats_batch = list()
    sensor2ego_trans_batch = list()
    img_metas_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_bda_mats,
            sweep_sensor2sensor_mats,
            sensor2ego_trans,
            img_metas,
        ) = iter_data[:8]
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        bda_mats_batch.append(sweep_bda_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        sensor2ego_trans_batch.append(sensor2ego_trans)
        img_metas_batch.append(img_metas)
    ret_list = [
        torch.stack(imgs_batch),  # 0
        torch.stack(sensor2ego_mats_batch),  # 1
        torch.stack(intrin_mats_batch),  # 2
        torch.stack(ida_mats_batch),  # 3
        torch.stack(bda_mats_batch),  # 4
        torch.stack(sensor2sensor_mats_batch),  # 5
        torch.stack(sensor2ego_trans_batch),  # 6
        img_metas_batch,  # 7
    ]
    result = dict()
    B, F, N, C, H, W = ret_list[0].shape  # noqa
    if F == 1:
        ret_list[0] = ret_list[0].view(B * N, C, H, W)
    result["img"] = ret_list[0]
    result["sensor2ego_mats"] = ret_list[1]
    result["intrin_mats"] = ret_list[2]
    result["ida_mats"] = ret_list[3]
    result["bda_mat"] = ret_list[4]
    result["sensor2sensor_mats"] = ret_list[5]
    result["sensor2ego_trans"] = ret_list[6]
    result["img_metas_batch"] = ret_list[7]
    result["cameras_paths"] = data[0][8]
    result["sensor2ego_rot"] = torch.stack([data[0][9]])
    return result
