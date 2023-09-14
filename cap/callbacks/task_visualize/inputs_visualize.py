# Copyright (c) Changan Auto. All rights reserved.
# 可视化分析分析模型输入数据
import copy
from typing import Callable

import torch
from torchvision.transforms import Resize

from cap.registry import OBJECT_REGISTRY
from ..callbacks import CallbackMixin

__all__ = ["InputsVisualize"]


class FakeImageId(object):
    def __init__(self, name):
        self.name = name

    def item(self):
        return self.name


def vehicle_heatmap_3d_detection_preprosses(batch):
    hm = batch['heatmap']
    dim = batch['dimensions']
    dep = 1.0 / (batch['depth'] + 1.0)
    dep = dep.logit()
    dep = torch.where(torch.isinf(dep), torch.full_like(dep, 0.), dep)
    loc_offset = batch['location_offset']
    fake_rot = loc_offset  # 模型输出的rot是个[bs,2,h,w]的张量，但是gt中只有解码后的rot，所以整个假的让解码器正常运行，然后再替换
    wh = batch['box2d_wh']  # 与实际量纲不同，但3D可视化中没有使用，先不管
    fake_pred = dict(
        hm=hm, dep=dep, rot=fake_rot, dim=dim, wh=wh, loc_offset=loc_offset
    )
    bs, _, h, w = batch['depth'].shape
    m = torch.zeros((bs, h * w, 1), dtype=batch['depth'].dtype)
    m[:, batch['index']] = batch['rotation_y']
    a = torch.zeros_like(m)
    a[:, batch['index']] = batch['alpha_x']
    fake_pred['real_rot_y'] = m.reshape((bs, 1, h, w))  # * torch.pi
    fake_pred['real_alpha_x'] = a.reshape((bs, 1, h, w))  # * torch.pi

    fake_data = dict(
        calib=batch['calib'],
        distCoeffs=torch.cat(batch['distCoeffs']),
        ori_img_shape=batch['orgin_wh'][0].flip(dims=[-1]),  # 由于只是显示模型输入效果，所以不需要返回原图
        # img=batch['img']  # 这个在解码时并未使用
    )

    param = dict()
    bs = batch['img'].size(0)
    param['ori_img'] = batch['ori_img']
    param['img'] = batch['img']
    param['ori_img_shape'] = batch['orgin_wh']
    param['img_name'] = batch['file_name']
    param['img_id'] = [FakeImageId(i) for i in batch['file_name']]
    param['img_height'] = batch['img_wh'][:, 1]
    param['img_width'] = batch['img_wh'][:, 0]
    param['calib'] = batch['calib']
    param['distCoeffs'] = torch.cat(batch['distCoeffs'])[None, ...]
    param['layout'] = ['hwc'] * bs
    param['color_space'] = ['bgr'] * bs
    return fake_pred, fake_data, param


@OBJECT_REGISTRY.register
class InputsVisualize(CallbackMixin):
    def __init__(self,
                 postprocess: Callable,
                 visualize,
                 **kwargs):
        super().__init__()
        self.visualize = visualize
        self.postprocess = postprocess  # decoder
        self.vis_num = kwargs.get('vis_num', 20)

    def on_batch_begin(self, batch, **kwargs):
        task_name = batch[-1]
        if task_name not in self.visualize.out_keys:
            return
        bs, _, h, w = batch[0]['depth'].shape
        if bs > 1:
            raise NotImplementedError("目前3D解码器Real3DDecoder只支持batch size=1的情况")
        try:
            pred, data, bts = eval(f"{task_name}_preprosses")(copy.deepcopy(batch[0]))
        except NameError:
            raise NotImplementedError(f"{task_name}任务的处理没有实现")
        model_outs = self.postprocess(pred, None, data, convert_task_sturct=True)
        if self.vis_num > 0:
            _, _, _ = self.visualize.visualize(None, bts, {task_name: model_outs})
            self.vis_num -= 1

    def __repr__(self):
        return "InputsVisualize"
