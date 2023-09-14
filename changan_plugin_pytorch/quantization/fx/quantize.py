from typing import Any, Dict, List, Optional, Tuple

import torch
from changan_plugin_pytorch.quantization.quantize import (
    convert,
    propagate_qconfig_,
)
from torch.fx import GraphModule
from torch.fx.graph import Node

from ..quantization_mappings import (
    get_qat_module_mappings,
    get_quantized_operator_mappings,
)
from .graph_module import (
    ObservedGraphModule,
    QuantizedGraphModule,
    is_observed_module,
)
from .pattern_utils import Pattern
from .qconfig_utils import (
    QConfigAny,
    convert_dict_to_ordered_dict,
    get_flattened_qconfig_dict,
)

# Define helper types
MatchResult = Tuple[Node, List[Node], Optional[Pattern], QConfigAny]


class Quantizer:
    def __init__(self):
        self.prepare_custom_config_dict: Dict[str, Any] = {}

    def save_state(self, observed: GraphModule) -> None:
        pass

    def restore_state(self, observed: GraphModule) -> None:
        assert is_observed_module(
            observed
        ), "incoming model must be produced by prepare_fx"
        pass

    def _qat_swap_modules(
        self,
        root: torch.nn.Module,
    ) -> None:
        convert(
            root,
            mapping=get_qat_module_mappings(),
            inplace=True,
            remove_qconfig=False,
        )

    def _prepare(
        self,
        model: GraphModule,
        qconfig_dict: Any,
        node_name_to_scope,
        prepare_custom_config_dict: Optional[Dict[str, Any]],
    ) -> ObservedGraphModule:
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        self.prepare_custom_config_dict = prepare_custom_config_dict

        convert_dict_to_ordered_dict(qconfig_dict)
        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)

        # set qconfig for modules
        propagate_qconfig_(model, flattened_qconfig_dict)

        self._qat_swap_modules(model)

        self.save_state(model)
        preserved_attributes = set(
            prepare_custom_config_dict.get("preserved_attributes", [])
        )
        model = ObservedGraphModule(model, model.graph, preserved_attributes)

        return model

    def prepare(
        self,
        model: GraphModule,
        qconfig_dict: Any,
        node_name_to_scope,
        prepare_custom_config_dict: Dict[str, Any] = None,
    ) -> ObservedGraphModule:
        return self._prepare(
            model,
            qconfig_dict,
            node_name_to_scope,
            prepare_custom_config_dict,
        )

    def _convert_swap_modules(
        self,
        root: torch.nn.Module,
        remove_qconfig,
    ) -> None:
        convert(
            root,
            mapping=get_quantized_operator_mappings(),
            inplace=True,
            remove_qconfig=remove_qconfig,
        )

    def _convert(
        self,
        model: GraphModule,
        convert_custom_config_dict: Dict[str, Any] = None,
        _remove_qconfig: bool = True,
    ) -> QuantizedGraphModule:
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        self.restore_state(model)

        self._convert_swap_modules(model, _remove_qconfig)

        self.save_state(model)
        preserved_attributes = set(
            convert_custom_config_dict.get("preserved_attributes", [])
        )
        model = QuantizedGraphModule(model, model.graph, preserved_attributes)

        return model

    def convert(
        self,
        model: GraphModule,
        convert_custom_config_dict: Dict[str, Any] = None,
        _remove_qconfig: bool = True,
    ) -> QuantizedGraphModule:
        quantized = self._convert(
            model,
            convert_custom_config_dict,
            _remove_qconfig=_remove_qconfig,
        )

        return quantized
