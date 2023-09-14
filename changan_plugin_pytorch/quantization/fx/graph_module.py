import copy
from typing import Any, Dict, Set, Union

import torch
from torch.fx import GraphModule
from torch.fx.graph import Graph


class GraphModuleWithAttr(GraphModule):
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: Set[str],
    ):
        self.preserved_attr_names = preserved_attr_names
        preserved_attrs = {
            attr: getattr(root, attr)
            for attr in self.preserved_attr_names
            if hasattr(root, attr)
        }
        super().__init__(root, graph)
        for attr in preserved_attrs:
            setattr(self, attr, preserved_attrs[attr])

    # GraphModule does not copy attributes which are not in the __dict__
    # of vanilla nn.Module.  So, we override __deepcopy__ in order
    # to copy the quantization specific attributes correctly.
    def __deepcopy__(self, memo):
        fake_mod = torch.nn.Module()
        fake_mod.__dict__ = copy.deepcopy(self.__dict__)
        return type(self)(fake_mod, self.graph, self.preserved_attr_names)


class FusedGraphModule(GraphModuleWithAttr):
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: Set[str],
    ):
        super(FusedGraphModule, self).__init__(
            root, graph, preserved_attr_names
        )


class ObservedGraphModule(GraphModuleWithAttr):
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: Set[str],
    ):
        super(ObservedGraphModule, self).__init__(
            root,
            graph,
            preserved_attr_names.union(set(["_prepare_custom_config_dict"])),
        )


def is_observed_module(module: Any) -> bool:
    return isinstance(module, ObservedGraphModule)


class QuantizedGraphModule(GraphModuleWithAttr):
    """This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    """

    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        preserved_attr_names: Set[str],
    ):
        super(QuantizedGraphModule, self).__init__(
            root, graph, preserved_attr_names
        )
