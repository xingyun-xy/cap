import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch
from changan_plugin_pytorch import nn as horizon_nn
from torch.fx.graph import Node
from torch.quantization.fuser_method_mappings import get_fuser_method

from ..fuse_modules import get_op_list_to_fuser_mapping
from .pattern_utils import MatchAllNode, register_fusion_pattern
from .quantization_types import QuantizerCls
from .utils import _parent_name

# ---------------------
# Fusion Pattern Registrations
# ---------------------


# Base Pattern Handler
class FuseHandler(ABC):
    """Base handler class for the fusion patterns"""

    def __init__(self, quantizer: QuantizerCls, node: Node):
        pass

    @abstractmethod
    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = None,
    ) -> Node:
        pass


@register_fusion_pattern((torch.nn.ReLU6, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.ReLU6, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU6, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.ReLU, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu6, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.functional.relu6, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu6, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.functional.relu, torch.nn.Conv3d))
@register_fusion_pattern((torch.nn.BatchNorm1d, torch.nn.Conv1d))
@register_fusion_pattern((torch.nn.BatchNorm2d, torch.nn.Conv2d))
@register_fusion_pattern((torch.nn.BatchNorm3d, torch.nn.Conv3d))
@register_fusion_pattern(
    (torch.nn.ReLU6, (torch.nn.BatchNorm1d, torch.nn.Conv1d))
)
@register_fusion_pattern(
    (torch.nn.ReLU6, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
)
@register_fusion_pattern(
    (torch.nn.ReLU6, (torch.nn.BatchNorm3d, torch.nn.Conv3d))
)
@register_fusion_pattern(
    (torch.nn.ReLU, (torch.nn.BatchNorm1d, torch.nn.Conv1d))
)
@register_fusion_pattern(
    (torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
)
@register_fusion_pattern(
    (torch.nn.ReLU, (torch.nn.BatchNorm3d, torch.nn.Conv3d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu6, (torch.nn.BatchNorm1d, torch.nn.Conv1d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu6, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu6, (torch.nn.BatchNorm3d, torch.nn.Conv3d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm1d, torch.nn.Conv1d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.Conv2d))
)
@register_fusion_pattern(
    (torch.nn.functional.relu, (torch.nn.BatchNorm3d, torch.nn.Conv3d))
)
@register_fusion_pattern(
    (
        torch.nn.quantized.FloatFunctional.add,
        torch.nn.Conv2d,
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        horizon_nn.quantized.FloatFunctional.add,
        torch.nn.Conv2d,
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        torch.nn.quantized.FloatFunctional.add,
        torch.nn.Conv3d,
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        horizon_nn.quantized.FloatFunctional.add,
        torch.nn.Conv3d,
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        torch.nn.quantized.FloatFunctional.add,
        (torch.nn.BatchNorm2d, torch.nn.Conv2d),
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        horizon_nn.quantized.FloatFunctional.add,
        (torch.nn.BatchNorm2d, torch.nn.Conv2d),
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        torch.nn.quantized.FloatFunctional.add,
        (torch.nn.BatchNorm3d, torch.nn.Conv3d),
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        horizon_nn.quantized.FloatFunctional.add,
        (torch.nn.BatchNorm3d, torch.nn.Conv3d),
        MatchAllNode,
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU6,
        (
            torch.nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm2d, torch.nn.Conv2d),
            MatchAllNode,
        ),
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU,
        (
            torch.nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm2d, torch.nn.Conv2d),
            MatchAllNode,
        ),
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU6,
        (
            horizon_nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm2d, torch.nn.Conv2d),
            MatchAllNode,
        ),
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU,
        (
            horizon_nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm2d, torch.nn.Conv2d),
            MatchAllNode,
        ),
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU6,
        (
            torch.nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm3d, torch.nn.Conv3d),
            MatchAllNode,
        ),
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU,
        (
            torch.nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm3d, torch.nn.Conv3d),
            MatchAllNode,
        ),
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU6,
        (
            horizon_nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm3d, torch.nn.Conv3d),
            MatchAllNode,
        ),
    )
)
@register_fusion_pattern(
    (
        torch.nn.ReLU,
        (
            horizon_nn.quantized.FloatFunctional.add,
            (torch.nn.BatchNorm3d, torch.nn.Conv3d),
            MatchAllNode,
        ),
    )
)
class ConvBNAddReLUFusion(FuseHandler):
    """
    This handler only fuse add with its first input node currently
    """

    def __init__(self, quantizer: QuantizerCls, node: Node):
        super().__init__(quantizer, node)
        from changan_plugin_pytorch import nn as horizon_nn

        self.node_list = []
        self.relu_node = None
        self.add_node = None
        self.bn_node = None

        # get relu node
        if (
            (
                node.op == "call_function"
                and node.target is torch.nn.functional.relu
            )
            or (
                node.op == "call_module"
                and type(quantizer.modules[node.target]) == torch.nn.ReLU
            )
            or (node.op == "call_method" and node.target == "relu")
        ):
            self.relu_node = node
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            self.relu_class = torch.nn.ReLU
        elif (
            (
                node.op == "call_function"
                and node.target is torch.nn.functional.relu6
            )
            or (
                node.op == "call_module"
                and type(quantizer.modules[node.target]) == torch.nn.ReLU6
            )
            or (node.op == "call_method" and node.target == "relu6")
        ):
            self.relu_node = node
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            self.relu_class = torch.nn.ReLU6

        # get add node
        if (
            node.op == "call_function"
            and node.target in (torch.add, operator.add)
        ) or (
            node.op == "call_method" and node.target == "add"
        ):  # Tensor.add or FloatFunctional.add
            self.add_node = node
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)

            if node.args[0].op == "get_attr":
                assert type(quantizer.modules[node.args[0].target]) in (
                    torch.nn.quantized.FloatFunctional,
                    horizon_nn.quantized.FloatFunctional,
                )
                node = node.args[1]  # skip get_attr node
            else:
                node = node.args[0]

        # get bn node
        assert node.op == "call_module"
        if type(quantizer.modules[node.target]) in [
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
        ]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            self.node_list.append(node)
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == "call_module"
        self.conv_node = node
        self.node_list.append(node)

        self.conv = quantizer.modules[self.conv_node.target]

    def fuse(
        self,
        quantizer: QuantizerCls,
        load_arg: Callable,
        fuse_custom_config_dict: Dict[str, Any] = {},
    ) -> Node:
        additional_fuser_method_mapping = get_op_list_to_fuser_mapping()
        additional_fuser_method_mapping.update(
            fuse_custom_config_dict.get("additional_fuser_method_mapping", {})
        )

        op_list = []
        if self.relu_node is not None:
            # since relu can be used multiple times, we'll need to
            # create a relu module for each match
            if self.relu_node.op == "call_module":
                relu = self.relu_class(
                    quantizer.modules[self.relu_node.target].inplace
                )
            else:
                inplace = False
                if len(self.relu_node.args) > 1:
                    inplace = self.relu_node.args[1]
                relu = self.relu_class(inplace)
            relu.training = self.conv.training
            op_list.append(relu)
        if self.add_node is not None:
            if self.add_node.op == "call_module":
                add = quantizer.modules[self.add_node.target]
            else:
                from changan_plugin_pytorch import nn as horizon_nn

                add = horizon_nn.quantized.FloatFunctional()
            add.training = self.conv.training
            op_list.append(add)
        if self.bn_node is not None:
            op_list.append(self.bn)
        op_list.append(self.conv)

        assert len(op_list) > 1

        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        fuser_method = get_fuser_method(
            op_type_list, additional_fuser_method_mapping
        )
        if fuser_method is None:
            raise NotImplementedError(
                "Cannot fuse modules: {}".format(op_type_list)
            )
        fused = fuser_method(*op_list)

        setattr(quantizer.modules[conv_parent_name], conv_name, fused)
        # self.node_list[0].replace_all_uses_with(self.conv_node)

        # if self.bn_node is not None:
        #     parent_name, name = _parent_name(self.bn_node.target)
        #     setattr(quantizer.modules[parent_name], name, torch.nn.Identity())  # noqa
        # relu may be used multiple times, so we don't set relu to identity

        if self.add_node is not None:
            new_node = torch.fx.Node(
                self.add_node.graph,
                self.add_node.name,
                "call_module",
                self.conv_node.target,
                (self.conv_node.args[0], self.add_node.args[-1]),
                {},
            )
        else:
            new_node = self.conv_node

        return quantizer.fused_graph.node_copy(new_node, load_arg)
