import numpy as np
import torch
from torch.nn import Conv2d


class ConvCompactor2d(torch.nn.Module):
    """A conv module which is added behind a normal conv module.

    This module is 1x1 conv with penalty gradients. The input and
    output channels are same as the attached conv module. After
    training, the filters of attached conv close to zero should be deleted
    by calculating the l2 norm of each filters in this module.

    Args:
        num_features (int): The out_channels of the normal conv block.
        conv_idx (int): The index of this compactor.
    """

    def __init__(self, num_features: int, conv_idx: int):
        super(ConvCompactor2d, self).__init__()
        self.conv_idx = conv_idx
        self.pwc = Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.pwc.weight.data.copy_(
            torch.eye(num_features, dtype=torch.float32).reshape(
                num_features, num_features, 1, 1
            )
        )

        self.register_buffer("mask", torch.ones(num_features))
        self.num_features = num_features

    def forward(self, inputs):
        return self.pwc(inputs)

    def set_mask(self, zero_indices):
        new_mask_value = torch.ones(self.num_features, dtype=torch.float32)
        new_mask_value[np.array(zero_indices)] = 0.0
        self.mask.copy_(new_mask_value)

    def get_num_mask_ones(self):
        mask_value = self.mask.cpu().numpy()
        return np.sum(mask_value == 1)

    def get_pwc_kernel_detach(self):
        return self.pwc.weight.detach()

    def get_metric_vector(self):
        metric_vector = (
            torch.sqrt(
                torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))
            )
            .cpu()
            .numpy()
        )
        return metric_vector

    def add_penalty_gradients(self):
        self.pwc.weight.grad.data = self.mask * self.pwc.weight.grad.data
        lasso_grad = self.pwc.weight.data * (
            (self.pwc.weight.data ** 2).sum(dim=(1, 2, 3), keepdim=True)
            ** (-0.5)
        )
        self.pwc.weight.grad.data.add_(1e-4, lasso_grad)
