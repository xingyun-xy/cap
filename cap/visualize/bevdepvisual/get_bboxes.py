import os
import os.path as osp
import tempfile

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import numba
import numpy as np
import torch

from cap.callbacks.task_visualize.visualize import BaseVisualize
from cap.data.datasets.bevdepth import Box, Quaternion
from cap.registry import OBJECT_REGISTRY
from projects.panorama.configs.resize.common import infer_save_prefix

__all__ = [
    "BevBBoxes",
]
n = 0


@OBJECT_REGISTRY.register
class BevBBoxes():

    def __init__(self, results):
        # self.pred = pred
        # self.img_metas = img_metas
        self.self_num_classes = [1, 2, 2, 1, 2, 2]
        self.self_norm_bbox = True
        self.self_max_num = 500
        self.self_out_size_factor = 4
        self.self_voxel_size = [0.2, 0.2, 8]
        self.self_pc_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
        self.self_score_threshold = 0.1
        self.self_post_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.self_test_cfg_nms_type = "circle"
        self.self_test_cfg_min_radius = [4, 12, 10, 1, 0.85, 0.175]
        self.self_test_cfg_post_max_size = 83
        self.self_bbox_coder_code_size = 9
        self.results = results
        super().__init__()

    def _format_bbox(self, results, img_metas, jsonfile_prefix=None):
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
        mapped_class_names = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]

        # print('Start to convert detection format...')

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

        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes, scores, labels = det
            boxes = boxes
            # print("\n")
            # print (img_metas)
            sample_token = img_metas[sample_id]["token"]
            # trans = np.array(img_metas[sample_id]["ego2global_translation"])
            # rot = Quaternion(img_metas[sample_id]["ego2global_rotation"])
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3].tolist()
                wlh = box[[4, 3, 5]].tolist()
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = Box(center, wlh, quat, velocity=box_vel)
                # nusc_box.rotate(rot)
                # nusc_box.translate(trans)
                if (np.sqrt(nusc_box.velocity[0]**2 + nusc_box.velocity[1]**2)
                        > 0.2):
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
                        attr = DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
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
            "meta": {
                "use_lidar": False,
                "use_camera": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": nusc_annos,
        }
        # mmcv.mkdir_or_exist(jsonfile_prefix)
        # res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        # print('Results writes to', res_path)
        # mmcv.dump(nusc_submissions, res_path)
        return nusc_submissions

    def format_results(
        self,
        results,
        img_metas,
        output_dir=None,  # json输出路径，自己指定
        result_names=["img_bbox"],  # 默认传值
        jsonfile_prefix=None,  # 默认传值
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
            if output_dir:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, output_dir)
                })
            else:
                result_files.update({
                    rasult_name:
                    self._format_bbox(results, img_metas, tmp_file_)
                })
        return result_files, tmp_dir

    def self_gather_feat(self, feats, inds, feat_masks=None):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor, optional): Mask of the feats.
                Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def self_topk(self, scores, K=80):
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int, optional): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = ((topk_inds.float() /
                    torch.tensor(width, dtype=torch.float)).int().float())
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self.self_gather_feat(topk_inds.view(batch, -1, 1),
                                          topk_ind).view(batch, K)
        topk_ys = self.self_gather_feat(topk_ys.view(batch, -1, 1),
                                        topk_ind).view(batch, K)
        topk_xs = self.self_gather_feat(topk_xs.view(batch, -1, 1),
                                        topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def self_transpose_and_gather_feat(self, feat, ind):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self.self_gather_feat(feat, ind)
        return feat

    def self_bbox_coder_decode(
        self,
        heat,
        rot_sine,
        rot_cosine,
        hei,
        dim,
        vel,
        reg=None,
        task_id=-1,
        self_post_center_range=None,
    ):
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1, W, H].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1, W, H].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 1, W, H].
            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self.self_topk(heat, K=self.self_max_num)

        if reg is not None:
            reg = self.self_transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.self_max_num, 2)
            xs = xs.view(batch, self.self_max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.self_max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.self_max_num, 1) + 0.5
            ys = ys.view(batch, self.self_max_num, 1) + 0.5

        # rotation value and direction label
        rot_sine = self.self_transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.self_max_num, 1)

        rot_cosine = self.self_transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.self_max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self.self_transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.self_max_num, 1)

        # dim of the box
        dim = self.self_transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.self_max_num, 3)

        # class label
        clses = clses.view(batch, self.self_max_num).float()
        scores = scores.view(batch, self.self_max_num)

        xs = (xs.view(batch, self.self_max_num, 1) *
              self.self_out_size_factor * self.self_voxel_size[0] +
              self.self_pc_range[0])
        ys = (ys.view(batch, self.self_max_num, 1) *
              self.self_out_size_factor * self.self_voxel_size[1] +
              self.self_pc_range[1])

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self.self_transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.self_max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.self_score_threshold is not None:
            thresh_mask = final_scores > self.self_score_threshold

        if self_post_center_range is not None:
            self_post_center_range = torch.tensor(self_post_center_range,
                                                  device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self_post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self_post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.self_score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    "bboxes": boxes3d,
                    "scores": scores,
                    "labels": labels,
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!")

        return predictions_dicts

    # @numba.jit(nopython=True)
    def circle_nms(self, dets, thresh, post_max_size=83):
        """Circular NMS.

        An object is only counted as positive if no other center
        with a higher confidence exists within a radius r using a
        bird-eye view distance metric.

        Args:
            dets (torch.Tensor): Detection results with the shape of [N, 3].
            thresh (float): Value of threshold.
            post_max_size (int, optional): Max number of prediction to be kept.
                Defaults to 83.

        Returns:
            torch.Tensor: Indexes of the detections to be kept.
        """

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        scores = dets[:, 2]
        order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
        ndets = dets.shape[0]
        suppressed = np.zeros((ndets), dtype=np.int32)
        keep = []
        for _i in range(ndets):
            i = order[_i]  # start with highest score box
            if (suppressed[i] == 1
                ):  # if any box have enough iou with this, remove it
                continue
            keep.append(i)
            for _j in range(_i + 1, ndets):
                j = order[_j]
                if suppressed[j] == 1:
                    continue
                # calculate center distance between i and j box
                dist = (x1[i] - x1[j])**2 + (y1[i] - y1[j])**2

                # ovr = inter / areas[j]
                if dist <= thresh:
                    suppressed[j] = 1

        if post_max_size < len(keep):
            return keep[:post_max_size]

        return keep

    # @numba.jit(nopython=True)
    # def size_aware_circle_nms(self, dets, thresh_scale, post_max_size=83):
    #     """Circular NMS.
    #     An object is only counted as positive if no other center
    #     with a higher confidence exists within a radius r using a
    #     bird-eye view distance metric.
    #     Args:
    #         dets (torch.Tensor): Detection results with the shape of [N, 3].
    #         thresh (float): Value of threshold.
    #         post_max_size (int): Max number of prediction to be kept. Defaults
    #             to 83
    #     Returns:
    #         torch.Tensor: Indexes of the detections to be kept.
    #     """
    #     x1 = dets[:, 0]
    #     y1 = dets[:, 1]
    #     dx1 = dets[:, 2]
    #     dy1 = dets[:, 3]
    #     yaws = dets[:, 4]
    #     scores = dets[:, -1]
    #     order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    #     ndets = dets.shape[0]
    #     suppressed = np.zeros((ndets), dtype=np.int32)
    #     keep = []
    #     for _i in range(ndets):
    #         i = order[_i]  # start with highest score box
    #         if suppressed[
    #                 i] == 1:  # if any box have enough iou with this, remove it
    #             continue
    #         keep.append(i)
    #         for _j in range(_i + 1, ndets):
    #             j = order[_j]
    #             if suppressed[j] == 1:
    #                 continue
    #             # calculate center distance between i and j box
    #             dist_x = abs(x1[i] - x1[j])
    #             dist_y = abs(y1[i] - y1[j])
    #             dist_x_th = (abs(dx1[i] * np.cos(yaws[i])) +
    #                          abs(dx1[j] * np.cos(yaws[j])) +
    #                          abs(dy1[i] * np.sin(yaws[i])) +
    #                          abs(dy1[j] * np.sin(yaws[j])))
    #             dist_y_th = (abs(dx1[i] * np.sin(yaws[i])) +
    #                          abs(dx1[j] * np.sin(yaws[j])) +
    #                          abs(dy1[i] * np.cos(yaws[i])) +
    #                          abs(dy1[j] * np.cos(yaws[j])))
    #             # ovr = inter / areas[j]
    #             if (dist_x <= dist_x_th * thresh_scale / 2
    #                     and dist_y <= dist_y_th * thresh_scale / 2):
    #                 suppressed[j] = 1
    #     return keep[:post_max_size]

    def gen_boxes(self):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        img_metas = self.results[1]
        preds = self.results[0][0]

        for task_id, preds_dict in enumerate(preds):

            preds_dict_ = {}

            preds_dict_["reg"] = preds_dict["reg"]
            preds_dict_["height"] = preds_dict["height"]
            preds_dict_["dim"] = preds_dict["dim"]
            preds_dict_["rot"] = preds_dict["rot"]
            preds_dict_["vel"] = preds_dict["vel"]
            preds_dict_["heatmap"] = preds_dict["heatmap"]  # .sigmoid()
            preds_dict = [preds_dict_]
            num_class_with_bg = self.self_num_classes[task_id]
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_heatmap = preds_dict[0]["heatmap"].sigmoid()

            batch_reg = preds_dict[0]["reg"]
            batch_hei = preds_dict[0]["height"]

            if self.self_norm_bbox:
                batch_dim = torch.exp(preds_dict[0]["dim"])
            else:
                batch_dim = preds_dict[0]["dim"]

            batch_rots = preds_dict[0]["rot"][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]["rot"][:, 1].unsqueeze(1)

            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"]
            else:
                batch_vel = None
            temp = self.self_bbox_coder_decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
                self_post_center_range=self.self_post_center_range,
            )
            assert self.self_test_cfg_nms_type in [
                "size_aware_circle",
                "circle",
                "rotate",
            ]
            batch_reg_preds = [box["bboxes"] for box in temp]
            batch_cls_preds = [box["scores"] for box in temp]
            batch_cls_labels = [box["labels"] for box in temp]
            if self.self_test_cfg_nms_type == "circle":
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]["bboxes"]
                    scores = temp[i]["scores"]
                    labels = temp[i]["labels"]
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        self.circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.self_test_cfg_min_radius[task_id],
                            post_max_size=self.self_test_cfg_post_max_size,
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == "bboxes":
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == "scores":
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == "labels":
                    flag = 0
                    for j, num_class in enumerate(self.self_num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels, img_metas[i]])

        # results = self.get_bboxes(input)

        # img_metas = list(np.load(ori_foler+"/results.npy", allow_pickle=True))[0][3]

        # ret_list[0].append(img_metas[0])

        all_pred_results = list()
        all_img_metas = list()
        for (validation_step_output) in (
                ret_list
        ):  # 这个result就是保存的结果，可以考虑一个sample一个sample地处理 也就是这个result是一个sample的结果
            all_pred_results.append(validation_step_output[:3])
            all_img_metas.append(validation_step_output[3])

        # os.makedirs("outputs", exist_ok=True)
        # direc = "tmp_vis_bev/"
        direc = infer_save_prefix + "/bev/"
        nusc_submissions = self._format_bbox(all_pred_results, all_img_metas)
        #self.format_results(all_pred_results, all_img_metas, output_dir = path) # 结果保存在outputs中
        # print("over!")
        # print(nusc_submissions)
        # print(all_img_metas[0])
        # from visualize_nusc import bev_vis
        os.makedirs(direc, exist_ok=True)
        path = direc
        #bev_vis(nusc_submissions,all_img_metas[0],path )

        # add ret_list for bev eval       modify by zmj
        return nusc_submissions, all_img_metas, path, ret_list

    def changan_gen_boxes(self):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        img_metas = self.results[1]
        preds = self.results[0][0]

        for task_id, preds_dict in enumerate(preds):

            preds_dict_ = {}

            if isinstance(preds_dict, list):
                preds_dict = preds_dict[0]

            preds_dict_["reg"] = preds_dict["reg"]
            preds_dict_["height"] = preds_dict["height"]
            preds_dict_["dim"] = preds_dict["dim"]
            preds_dict_["rot"] = preds_dict["rot"]
            preds_dict_["vel"] = preds_dict["vel"]
            preds_dict_["heatmap"] = preds_dict["heatmap"]  # .sigmoid()
            preds_dict = [preds_dict_]
            num_class_with_bg = self.self_num_classes[task_id]
            batch_size = preds_dict[0]["heatmap"].shape[0]
            batch_heatmap = preds_dict[0]["heatmap"].sigmoid()

            batch_reg = preds_dict[0]["reg"]
            batch_hei = preds_dict[0]["height"]

            if self.self_norm_bbox:
                batch_dim = torch.exp(preds_dict[0]["dim"])
            else:
                batch_dim = preds_dict[0]["dim"]

            batch_rots = preds_dict[0]["rot"][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]["rot"][:, 1].unsqueeze(1)

            if "vel" in preds_dict[0]:
                batch_vel = preds_dict[0]["vel"]
            else:
                batch_vel = None
            temp = self.self_bbox_coder_decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
                self_post_center_range=self.self_post_center_range,
            )
            assert self.self_test_cfg_nms_type in [
                "size_aware_circle",
                "circle",
                "rotate",
            ]
            batch_reg_preds = [box["bboxes"] for box in temp]
            batch_cls_preds = [box["scores"] for box in temp]
            batch_cls_labels = [box["labels"] for box in temp]
            if self.self_test_cfg_nms_type == "circle":
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]["bboxes"]
                    scores = temp[i]["scores"]
                    labels = temp[i]["labels"]
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        self.circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.self_test_cfg_min_radius[task_id],
                            post_max_size=self.self_test_cfg_post_max_size,
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == "bboxes":
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == "scores":
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == "labels":
                    flag = 0
                    for j, num_class in enumerate(self.self_num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])

        # img_metas = list(np.load(ori_foler+"/results.npy", allow_pickle=True))[0][3]

        ret_list[0].append(img_metas[0])

        all_pred_results = list()
        all_img_metas = list()
        for (validation_step_output) in (
                ret_list
        ):  # 这个result就是保存的结果，可以考虑一个sample一个sample地处理 也就是这个result是一个sample的结果
            all_pred_results.append(validation_step_output[:3])
            all_img_metas.append(validation_step_output[3])

        # os.makedirs("outputs", exist_ok=True)
        direc = infer_save_prefix + "/bev/"
        # nusc_submissions = self._format_bbox(all_pred_results, all_img_metas)
        # self.format_results(all_pred_results, all_img_metas, output_dir = path) # 结果保存在outputs中
        print("over!")
        # print(nusc_submissions)
        # print(all_img_metas[0])
        # from visualize_nusc import bev_vis
        os.makedirs(direc, exist_ok=True)
        path = direc
        #bev_vis(nusc_submissions,all_img_metas[0],path )

        return ret_list, img_metas, path,
