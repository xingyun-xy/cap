# Copyright (c) Changan Auto. All rights reserved.
import torch
import torch.nn as nn


class GroupNorm2d(nn.GroupNorm):
    """GN Module for GN Fusion.

    The difference between nn.GroupNorm and GroupNorm2d is that\
    GroupNorm2d saves the running_mean and running_var for GroupNorm Fusion.
    Besides, there is a new use_momentum option for saving running_mean and\
    running_var.

    Args:
        num_groups: Number of groups.
        num_channels: The channels number of the inputs tensor.
        eps: a value added to the denominator for numerical stability.
        momentum: the value used for the running_mean\
        and running_var computation.
        affine: If set to True, this module has learnable affine parameters.
        track_running_stats: If set to True, this module tracks\
        the running mean and variance.
        use_momentum: If using momentum to update running_mean and running_var.
        The usage is the same as the nn.GroupNorm.
    """

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        use_momentum=False,
    ):
        super(GroupNorm2d, self).__init__(
            num_groups, num_channels, eps, affine
        )
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.use_momentum = use_momentum

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_groups))
            self.register_buffer("running_var", torch.ones(num_groups))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[operator]
            self.running_var.fill_(1)  # type: ignore[operator]
            self.num_batches_tracked.zero_()  # type: ignore[operator]

    def forward(self, inputs):

        # calculate running estimates
        N, C, H, W = inputs.shape
        G = self.num_groups
        inputs = inputs.reshape([N, G, C // G, H, W])
        means = inputs.mean([2, 3, 4], keepdim=True)
        # use biased var in train Bessel's Correction by unbiased=False,
        vars = inputs.var([2, 3, 4], unbiased=False, keepdim=True)

        if self.track_running_stats:

            current_means = means.mean(dim=0, keepdim=True).reshape(
                self.num_groups
            )
            current_vars = vars.mean(dim=0, keepdim=True).reshape(
                self.num_groups
            )

            if self.use_momentum:
                exponential_average_factor = 0.0
                if self.training:
                    if self.num_batches_tracked is not None:
                        self.num_batches_tracked += 1
                        if (
                            self.momentum is None
                        ):  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(
                                self.num_batches_tracked
                            )
                        else:  # use exponential moving average
                            exponential_average_factor = self.momentum

                n = inputs.numel() / (inputs.size(0) * inputs.size(1))
                current_means = (
                    exponential_average_factor * current_means
                    + (1 - exponential_average_factor) * self.running_mean
                )

                # update running_var with unbiased var
                current_vars = (
                    exponential_average_factor * current_vars * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var
                )

            self.running_mean.copy_(current_means)
            self.running_var.copy_(current_vars)

        inputs = (inputs - means) / (torch.sqrt(vars + self.eps))
        inputs = inputs.reshape([N, C, H, W])
        if self.affine:
            inputs = (
                inputs * self.weight[None, :, None, None]
                + self.bias[None, :, None, None]
            )

        return inputs
