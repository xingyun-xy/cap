import torch
from changan_plugin_pytorch.nn.quantized import FloatFunctional
from changan_plugin_pytorch.utils.fx_helper import get_supported_method
from torch.fx import GraphModule
from torch.fx.node import Node
from typing import Dict, Tuple


__all__ = ["swap_ff_with_horizonff", "replace_function_with_functional"]


def _parent_name(target):
    """
    Get the parent name and current module name from a qualified name
    """
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


def replace_function_with_functional(
    model: GraphModule, node_name_to_scope: Dict[str, Tuple[str, type]]
):
    """
    Replace function type operations in a model
    with corresponding FloatFunctional

    Args:
        model (GraphModule): The input model
        node_name_to_scope (Dict[str, Tuple[str, type]]): Mapping from node
            name to the owner module name and type.
    """
    supported_functional = get_supported_method()[FloatFunctional]

    for node in list(model.graph.nodes):
        node: Node
        if (
            node.op in ("call_function", "call_method")
            and (
                node.target in supported_functional
                or node.target.__name__ in supported_functional
            )
            and node.args[0].op != "get_attr"  # already FloatFunctional
        ):
            if node.target in supported_functional:
                op_name = node.target
            else:
                op_name = node.target.__name__

            # add FloatFunctional to current module
            current_scope_name = node_name_to_scope[node.name][0]

            module_idx = 0
            module_name = "{}_generated_{}_{}".format(
                current_scope_name, op_name, module_idx
            )
            while hasattr(model, module_name):
                module_idx += 1
                module_name = "{}_generated_{}_{}".format(
                    current_scope_name, op_name, module_idx
                )

            setattr(model, module_name, FloatFunctional())

            # modify graph
            model.graph.inserting_before(node)
            get_attr_node = model.graph.get_attr(module_name)
            call_method_node = model.graph.call_method(
                op_name, args=(get_attr_node,) + node.args
            )
            node.replace_all_uses_with(call_method_node)

    # delete the nodes of function
    model.graph.eliminate_dead_code()
