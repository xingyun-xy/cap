# Copyright (c) Changan Auto. All rights reserved.

from typing import Optional, Sequence

import torch

from cap.registry import OBJECT_REGISTRY
from cap.visualize import vis_parallax
from .metric import EvalMetric

__all__ = ["ElevationMetric"]


def compute_depth_errors(gt: torch.Tensor, pred: torch.Tensor):
    """Calculate multi metric for depth.

    The a1, a2, a3, rmse, rmse_log, abs_rel and sq_rel
    metrics would be caculated for depth.

    Args:
        gt: gt depth.
        pred: pred depth.
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_height_errors(
    h_gt: torch.Tensor,
    h_pred: torch.Tensor,
    d_gt: torch.Tensor,
    d_pred: torch.Tensor,
    h_range: Optional[Sequence] = (0, 0.1, 0.3, 0.5, 1.0, 2.0),
    d_range: Optional[Sequence] = (0, 10, 30, 50, 80, 100),
    height_range: Optional[Sequence] = (-0.2, 2),
    call_range_errs: bool = True,
):
    """Calculate multi range metric for height.

    Dfferent tasks need to focus on different areas, in the height
    metric, the errors are split into multiple depth and height range.

    Args:
        h_gt: gt height.
        h_pred: pred height.
        d_gt: gt depth.
        d_pred: pred depth.
        h_range: bins of height to eval.
        d_range: bins of depth to eval.
        height_range: gt range to eval.
        call_range_errs: whether call range errors.
    """
    HEIGHT_MIN, HEIGHT_MAX = height_range
    h_pred = h_pred.clip(HEIGHT_MIN, HEIGHT_MAX)
    detal = (h_gt - h_pred).abs()
    h_errs_range = h_pred.new_zeros((len(d_range) - 1, len(h_range) - 1))
    h_errs = h_pred.new_zeros((len(d_range) - 1, len(h_range) - 1))
    for i in range(1, len(d_range)):
        for j in range(1, len(h_range)):
            if call_range_errs:
                mask_d = (d_gt > d_range[i - 1]) & (d_gt <= d_range[i])
                mask_h = (h_gt > h_range[j - 1]) & (h_gt <= h_range[j])
                mask = mask_d & mask_h
                if mask.sum() != 0:
                    h_err = ((detal * mask).sum() / mask.sum()).item()
                else:
                    h_err = 0.0
                h_errs_range[i - 1][j - 1] = h_err

            mask_d = (d_gt > d_range[0]) & (d_gt <= d_range[i])
            mask_h = (h_gt > h_range[0]) & (h_gt <= h_range[j])
            mask = mask_d & mask_h
            if mask.sum() != 0:
                h_err = ((detal * mask).sum() / mask.sum()).item()
            else:
                h_err = 0.0
            h_errs[i - 1][j - 1] = h_err
    return h_errs_range, h_errs


def compute_gamma_errors(
    gamma_gt: torch.Tensor,
    gamma_pred: torch.Tensor,
    gamma_range: Optional[Sequence] = (-0.06, 0.31),
):
    """Calculate metric for gamma.

    Calculate L1 error for gamma with a mask.

    Args:
        gamma_gt: gt gamam.
        gamma_pred: pred gamma.
        gamma_range: gt range to eval.
    """
    GAMMA_MIN, GAMMA_MAX = gamma_range
    gamma_pred = gamma_pred.clip(GAMMA_MIN, GAMMA_MAX)
    gamma_err = (gamma_gt - gamma_pred).abs()
    mask = (gamma_gt != -1) & (gamma_gt > GAMMA_MIN) & (gamma_gt < GAMMA_MAX)
    abs_gamma = (gamma_err * mask).sum() / mask.sum()
    return abs_gamma


@OBJECT_REGISTRY.register
class ElevationMetric(EvalMetric):
    """Calculate multi distance metric for elevation task.

    ElevationMetric include gamma, depth and height metric.

    Args:
        metrics: the gt types to eval.
        name: metric name.
        gamma_scale: scale of gamma gt.
    """

    def __init__(
        self,
        metrics: Optional[Sequence] = None,
        gamma_scale: float = 1000.0,
        vis: bool = False,
        save_path: str = "",
    ):
        self.gamma_scale = gamma_scale
        assert all(
            ["depth" == i or "height" == i or "gamma" == i for i in metrics]
        ), "metrics should only contain elements of 'gamma', 'depth', 'height'"
        self.metrics = metrics
        self.vis = vis
        self.save_path = save_path

        DIS_TYPE = {
            "depth": ["depth_error"],
            "height": ["h_errs_range", "h_errs_range"],
            "gamma": ["abs_gamma"],
        }
        self.name = []
        for metric in metrics:
            self.name += DIS_TYPE[metric]
        super(ElevationMetric, self).__init__(self.name)

    def _init_states(self):
        if "depth" in self.metrics:
            self.add_state(
                name="depth_error",
                default=torch.zeros(7),
                dist_reduce_fx="sum",
            )
        if "height" in self.metrics:
            self.add_state(
                name="h_errs_range",
                default=torch.zeros(5, 5),
                dist_reduce_fx="sum",
            )
            self.add_state(
                name="h_errs",
                default=torch.zeros(5, 5),
                dist_reduce_fx="sum",
            )
        if "gamma" in self.metrics:
            self.add_state(
                name="abs_gamma",
                default=torch.zeros(1),
                dist_reduce_fx="sum",
            )

        self.add_state(
            name="num_inst",
            default=torch.zeros(1),
            dist_reduce_fx="sum",
        )

    def parallax_metrics(
        self,
        depth_gt: torch.Tensor,
        depth: torch.Tensor,
        depth_range: Optional[Sequence] = (0.1, 80),
    ):
        """Calculate metric for parallax.

        Args:
            depth_gt: gt depth.
            depth: pred depth.
            depth_range: gt range to eval.
        """
        DEPTH_MIN, DEPTH_MAX = depth_range
        depth_gt = depth_gt
        depth = depth
        depth = depth.clip(DEPTH_MIN, DEPTH_MAX)
        valid_depth_points = (depth_gt >= DEPTH_MIN) * (depth_gt <= DEPTH_MAX)
        valid_depth_points = valid_depth_points.to(dtype=torch.bool)

        depth_gt_list = depth_gt[valid_depth_points]
        depth_list = depth[valid_depth_points]

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(
            depth_gt_list, depth_list
        )

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def update(
        self,
        gamma_gt,
        depth_gt,
        height_gt,
        gamma_pred,
        depth_pred,
        height_pred,
        color_img_cur=None,
        timestamp=None,
    ):
        if self.vis:
            timestamp = str(timestamp[:1].cpu().squeeze().numpy())
        if "depth" in self.metrics:
            (
                abs_rel,
                sq_rel,
                rmse,
                rmse_log,
                a1,
                a2,
                a3,
            ) = self.parallax_metrics(depth_gt, depth_pred)
            depth_error_i = torch.stack(
                [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3], axis=0
            )
            self.depth_error += depth_error_i
            if self.vis:
                vis_parallax(
                    depth_pred[:1],
                    color_img_cur[:1],
                    save=True,
                    path=self.save_path,
                    timestamp=timestamp,
                    vis_type="depth",
                )
        if "height" in self.metrics:
            h_errs_range_i, h_errs_i = compute_height_errors(
                height_gt, height_pred, depth_gt, depth_pred
            )
            self.h_errs_range += h_errs_range_i
            self.h_errs += h_errs_i
            if self.vis:
                vis_parallax(
                    height_pred[:1],
                    color_img_cur[:1],
                    save=True,
                    path=self.save_path,
                    timestamp=timestamp,
                    vis_type="height",
                )
        if "gamma" in self.metrics:
            gamma_pred = gamma_pred / self.gamma_scale  # [1, 1, 512, 960]
            gamma_gt = gamma_gt / self.gamma_scale
            abs_gamma_i = compute_gamma_errors(gamma_gt, gamma_pred)
            self.abs_gamma += abs_gamma_i
            if self.vis:
                vis_parallax(
                    gamma_pred[:1],
                    color_img_cur[:1],
                    save=True,
                    path=self.save_path,
                    timestamp=timestamp,
                    vis_type="gamma",
                )
        self.num_inst += 1

    def compute(self):
        val = []
        if "depth" in self.metrics:
            val.append(self.depth_error / self.num_inst)
        if "height" in self.metrics:
            val.append(self.h_errs_range / self.num_inst)
            val.append(self.h_errs / self.num_inst)
        if "gamma" in self.metrics:
            val.append(self.abs_gamma / self.num_inst)
        return val
