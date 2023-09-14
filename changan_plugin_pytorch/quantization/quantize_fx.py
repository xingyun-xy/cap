from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from changan_plugin_pytorch.utils.fx_helper import CustomTracer
from changan_plugin_pytorch.utils.model_helper import swap_ff_with_horizonff
from torch.fx import GraphModule
from torch.fx.node import Argument, Node, Target
from torch.nn.intrinsic import _FusedModule

from .fx import Fuser, Quantizer
from .fx.graph_module import ObservedGraphModule, QuantizedGraphModule
from .fx.utils import replace_function_with_functional
from .quantization_mappings import get_qat_module_mappings


def _check_is_graph_module(model: torch.nn.Module) -> None:
    if not isinstance(model, GraphModule):
        raise ValueError(
            "input model must be a GraphModule, "
            + "Got type:"
            + str(type(model))
            + " Please make "
            + "sure to follow the tutorials."
        )


def _fuse_fx(
    graph_module: GraphModule, fuse_custom_config_dict: Dict[str, Any] = None
) -> GraphModule:
    r"""
    Internal helper function to fuse modules inpreparation for quantization

    Args:
        graph_module:
            GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
    _check_is_graph_module(graph_module)

    fuser = Fuser()
    return fuser.fuse(graph_module, fuse_custom_config_dict)


class Scope(object):
    """Scope object that records the module path and the module type
    of a module. Scope is used to track the information of the module
    that contains a Node in a Graph of GraphModule. For example:
    class Sub(torch.nn.Module):
        def forward(self, x):
            # This will be a call_method Node in GraphModule,
            # scope for this would be (module_path="sub", module_type=Sub)
            return x.transpose(1, 2)

    class M(torch.nn.Module):
        def __init__(self):
            self.sub = Sub()

        def forward(self, x):
            # This will be a call_method Node as well,
            # scope for this would be (module_path="", None)
            x = x.transpose(1, 2)
            x = self.sub(x)
            return x

    """

    def __init__(self, module_path: str, module_type: Any):
        super().__init__()
        self.module_path = module_path
        self.module_type = module_type


class ScopeContextManager(object):
    """A context manager to track the Scope of Node during symbolic
    tracing.
    When entering a forward function of a Module, we'll update the scope
    information of the current module, and when we exit, we'll restore
    the previous scope information.
    """

    def __init__(
        self,
        scope: Scope,
        current_module: torch.nn.Module,
        current_module_path: str,
    ):
        super().__init__()
        self.prev_module_type = scope.module_type
        self.prev_module_path = scope.module_path
        self.scope = scope
        self.scope.module_path = current_module_path
        self.scope.module_type = type(current_module)

    def __enter__(self):
        return

    def __exit__(self, *args):
        self.scope.module_path = self.prev_module_path
        self.scope.module_type = self.prev_module_type
        return


class QuantizationTracer(CustomTracer):
    def __init__(
        self,
        skipped_module_names: List[str],
        skipped_module_classes: List[Callable],
    ):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type
        # of top level module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}

    def is_leaf_module(
        self, m: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        return (
            (
                m.__module__.startswith("torch.nn")
                and not isinstance(m, torch.nn.Sequential)
                and not isinstance(m, torch.nn.ModuleList)
                and not isinstance(m, torch.nn.ModuleDict)
            )
            or module_qualified_name in self.skipped_module_names
            or type(m) in self.skipped_module_classes
            or isinstance(m, _FusedModule)
            or super().is_leaf_module(m, module_qualified_name)
        )

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        module_qualified_name = self.path_of_module(m)
        # Creating scope with information of current module
        # scope will be restored automatically upon exit
        with ScopeContextManager(self.scope, m, module_qualified_name):
            return super().call_module(m, forward, args, kwargs)

    def create_node(
        self,
        kind: str,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (
            self.scope.module_path,
            self.scope.module_type,
        )
        return node


def _prepare_fx(
    model: torch.nn.Module,
    qconfig_dict: Any,
    prepare_custom_config_dict: Dict[str, Any] = None,
) -> ObservedGraphModule:
    r"""Internal helper function for prepare_fx
    Args:
        `model`, `qconfig_dict`, `prepare_custom_config_dict`:
            see docs for :func:`~torch.quantization.prepare_fx`
    """
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}

    swap_ff_with_horizonff(model)

    preserved_attributes = prepare_custom_config_dict.get(
        "preserved_attributes", []
    )
    tracer = QuantizationTracer([], list(get_qat_module_mappings().keys()))
    graph_module = GraphModule(model, tracer.trace(model))
    for attr_name in preserved_attributes:
        setattr(graph_module, attr_name, getattr(model, attr_name))

    replace_function_with_functional(graph_module, tracer.node_name_to_scope)

    graph_module = _fuse_fx(graph_module, prepare_custom_config_dict)

    quantizer = Quantizer()
    prepared = quantizer.prepare(
        graph_module,
        qconfig_dict,
        tracer.node_name_to_scope,
        prepare_custom_config_dict=prepare_custom_config_dict,
    )

    for attr_name in preserved_attributes:
        setattr(prepared, attr_name, getattr(model, attr_name))
    return prepared


def fuse_fx(
    model: torch.nn.Module, fuse_custom_config_dict: Dict[str, Any] = None
) -> GraphModule:
    r"""
    Fuse modules like conv+add+bn+relu etc, model must be in eval mode.
    Fusion rules are defined in
    changan_plugin_pytorch.quantization.fx.fusion_pattern.py

    Args:
        `model`: a torch.nn.Module model
        `fuse_custom_config_dict`:
            Dictionary for custom configurations for fuse_fx, e.g.
            fuse_custom_config_dict = {
            # Attributes that are not used in forward function will
            # be removed when constructing GraphModule, this is a list of
            # attributes to preserve as an attribute of the GraphModule
            # even when they are not used in the code, these attributes
            # will also persist through deepcopy
            "preserved_attributes": ["preserved_attr"],
            }

    Example:
    ```python
    from torch.quantization import fuse_fx
    m = Model().eval()
    m = fuse_fx(m)
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.fuse_fx")

    swap_ff_with_horizonff(model)

    tracer = QuantizationTracer(
        [], skipped_module_classes=list(get_qat_module_mappings().keys())
    )
    graph_module = GraphModule(model, tracer.trace(model))
    preserved_attributes: Set[str] = set()
    if fuse_custom_config_dict:
        preserved_attributes = set(
            fuse_custom_config_dict.get("preserved_attributes", [])
        )
    for attr_name in preserved_attributes:
        setattr(graph_module, attr_name, getattr(model, attr_name))

    replace_function_with_functional(graph_module, tracer.node_name_to_scope)

    return _fuse_fx(graph_module, fuse_custom_config_dict)


def prepare_qat_fx(
    model: Union[torch.nn.Module, GraphModule],
    qconfig_dict: Any,
    prepare_custom_config_dict: Dict[str, Any] = {},
) -> ObservedGraphModule:
    r"""Prepare a model for quantization aware training
    Args:
      `model`: torch.nn.Module model or GraphModule model (maybe from fuse_fx),
               must be in train mode
      `qconfig_dict`:
        qconfig_dict is a dictionary with the followingconfigurations:
      qconfig_dict = {
        # optional, global config
        "": qconfig?,

        # optional, used for module and function types
        # could also be split into module_types and function_types if we prefer
        "object_type": [
            (torch.nn.Conv2d, qconfig?),
            (torch.nn.functional.add, qconfig?),
            ...,
        ],

        # optional, matched in order, first match takes precedence
        "module_name_regex": [
            ("foo.*bar.*conv[0-9]+", qconfig?)
            ...,
        ],

        # optional, used for module names
        "module_name": [
            ("foo.bar", qconfig?)
            ...,
        ],
        # priority (in increasing order):
        #   global, object_type, module_name, module.qconfig
        # qconfig == None means fusion and quantization should be
        # skipped for anything matching the rule
        }
        `prepare_custom_config_dict`: customization configuration dictionary
        for quantization tool:
        prepare_custom_config_dict = {
            # Attributes that are not used in forward function will
            # be removed when constructing GraphModule, this is a list of
            # attributes to preserve as an attribute of the GraphModule even
            # when they are not used in the code, these attributes will also
            # persist through deepcopy
            "preserved_attributes": ["preserved_attr"],
        }

    Return:
      A GraphModule with fake quant modules (configured by qconfig_dict),
      ready for quantization aware training

    Example:
    ```python
    import torch
    from torch.quantization import get_default_qat_qconfig
    from torch.quantization import prepare_fx

    qconfig = get_default_qat_qconfig('fbgemm')
    def train_loop(model, train_data):
        model.train()
        for image, target in data_loader:
            ...

    float_model.train()
    qconfig_dict = {"": qconfig}
    prepared_model = prepare_fx(float_model, qconfig_dict)
    # Run calibration
    train_loop(prepared_model, train_loop)
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.prepare_qat_fx")

    msg = "prepare_custom_config_dict only cound have 'preserved_attributes'"
    assert len(prepare_custom_config_dict) <= 1, msg
    if len(prepare_custom_config_dict) > 0:
        assert "preserved_attributes" in prepare_custom_config_dict, msg

    return _prepare_fx(model, qconfig_dict)


def _convert_fx(
    graph_module: GraphModule,
    convert_custom_config_dict: Dict[str, Any] = {},
    _remove_qconfig: bool = True,
) -> QuantizedGraphModule:
    _check_is_graph_module(graph_module)

    quantizer = Quantizer()
    quantized = quantizer.convert(
        graph_module,
        convert_custom_config_dict,
        _remove_qconfig=_remove_qconfig,
    )

    preserved_attributes = convert_custom_config_dict.get(
        "preserved_attributes", []
    )
    for attr_name in preserved_attributes:
        setattr(quantized, attr_name, getattr(graph_module, attr_name))
    return quantized


def convert_fx(
    graph_module: GraphModule,
    convert_custom_config_dict: Dict[str, Any] = {},
    _remove_qconfig: bool = True,
) -> QuantizedGraphModule:
    r"""Convert a calibrated or trained model to a quantized model
    Args:
        `graph_module`: A prepared and calibrated/trained model (GraphModule)
        `convert_custom_config_dict`:
            dictionary for custom configurations for convert function:
        convert_custom_config_dict = {
          # Attributes that are not used in forward function will
          # be removed when constructing GraphModule, this is a list of
          # attributes to preserve as an attribute of the GraphModule even
          # when they are not used in the code
          "preserved_attributes": ["preserved_attr"],
        }
        `_remove_qconfig`:
            Option to remove the qconfig attributes in the model after convert.

    Return:
        A quantized model (GraphModule)

    Example:
    ```python
    # prepared_model: the model after prepare_fx/prepare_qat_fx and
    # calibration/training
    quantized_model = convert_fx(prepared_model)
    ```
    """
    torch._C._log_api_usage_once("quantization_api.quantize_fx.convert_fx")

    msg = "convert_custom_config_dict only cound have 'preserved_attributes'"
    assert len(convert_custom_config_dict) <= 1, msg
    if len(convert_custom_config_dict) > 0:
        assert "preserved_attributes" in convert_custom_config_dict, msg

    return _convert_fx(
        graph_module,
        convert_custom_config_dict,
        _remove_qconfig=_remove_qconfig,
    )
