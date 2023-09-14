# Copyright (c) Changan Auto. All rights reserved.

from typing import Mapping

import torch

from cap.registry import OBJECT_REGISTRY

__all__ = ["ArgmaxPostprocess"]


# TODO(mengao.zhao, HDLT-235): refactor argmax_postprocess #
@OBJECT_REGISTRY.register
class ArgmaxPostprocess(torch.nn.Module):
    """Apply argmax of data in pred_dict.

    Args:
        data_name (str): name of data to apply argmax.
        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not.

    """

    def __init__(self, data_name: str, dim: int, keepdim: bool = False):
        super(ArgmaxPostprocess, self).__init__()
        self.data_name = data_name
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, pred_dict: Mapping, *args):
        if isinstance(pred_dict[self.data_name], list):
            argmax_datas = []
            for each_data in pred_dict[self.data_name]:
                argmax_datas.append(each_data.argmax(self.dim, self.keepdim))
            pred_dict[self.data_name] = argmax_datas
        elif isinstance(pred_dict[self.data_name], torch.Tensor):
            pred_dict[self.data_name] = pred_dict[self.data_name].argmax(
                self.dim, self.keepdim
            )
        else:
            raise TypeError("only support torch.tensor or list[torch.tensor]")
        return pred_dict

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
