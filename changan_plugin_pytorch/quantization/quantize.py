"""
prepare and convert
"""
import copy
import warnings
from functools import partial
from typing import Dict, Optional

import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn.qat.stubs import QuantStub
from changan_plugin_pytorch.qtensor import qtensor_allow_float_operation
from changan_plugin_pytorch.utils.fx_helper import CustomTracer
from torch import fx
from torch.fx.immutable_collections import immutable_list
from torch.quantization.quantize import _remove_qconfig, get_unique_devices_

from .observer import CalibObserver, FixedScaleObserver
from .quantization_mappings import (
    get_qat_module_mappings,
    get_qconfig_propagation_list,
    get_quantized_operator_mappings,
    wrap_qat_modules_for_fx,
)


def after_calibration(model):
    from .fake_quantize import CalibFakeQuantize

    for m in model.modules():
        if isinstance(m, CalibFakeQuantize):
            return True
    return False


def remove_qconfig(model):
    for m in model.modules():
        if hasattr(m, "qconfig"):
            if m.qconfig is None:
                continue
            if m.qconfig.activation is None or m.qconfig.weight is None:
                continue
            else:
                del m.qconfig


def _set_qat_activation_post_process_state(m):
    from ..nn.qat import Conv2d, ConvTranspose2d

    fake_quant_enabled = m.activation_post_process.fake_quant_enabled.item()
    observer_enabled = m.activation_post_process.observer_enabled.item()
    if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d):
        m.activation_post_process = m.qconfig.activation(
            channel_len=m.out_channels
        )
    else:
        m.activation_post_process = m.qconfig.activation()
    if fake_quant_enabled != 1:
        m.activation_post_process.disable_fake_quant()
    if observer_enabled != 1:
        m.activation_post_process.disable_observer()


def _calculate_statistical_qparams(m):
    amax = m.activation_post_process.activation_post_process.compute_amax()
    _set_qat_activation_post_process_state(m)
    observer = m.activation_post_process.activation_post_process
    if isinstance(observer, FixedScaleObserver):
        warnings.warn("use FixedScaleObserver in qat but not in calibration")
    if amax is not None and not isinstance(observer, FixedScaleObserver):
        observer.min_val.resize_(amax.shape)
        observer.min_val.copy_(-amax)
        observer.max_val.resize_(amax.shape)
        observer.max_val.copy_(amax)

    (
        scale,
        zero_point,
    ) = observer.calculate_qparams()
    return scale, zero_point


def _get_fixed_qparams(m):
    (
        scale,
        zero_point,
    ) = m.activation_post_process.activation_post_process.calculate_qparams()
    return scale, zero_point


def _calculate_activation_qparams(m):

    setted_scale = m.activation_post_process.scale
    if isinstance(
        m.activation_post_process.activation_post_process, CalibObserver
    ):
        scale, zero_point = _calculate_statistical_qparams(m)
    else:
        assert isinstance(
            m.activation_post_process.activation_post_process,
            FixedScaleObserver,
        )
        scale, zero_point = _get_fixed_qparams(m)
        fixed_scale_observer = (
            m.activation_post_process.activation_post_process
        )
        _set_qat_activation_post_process_state(m)
        if isinstance(
            m.activation_post_process.activation_post_process,
            FixedScaleObserver,
        ):
            qat_fixed_scale = (
                m.activation_post_process.activation_post_process.scale
            )
            assert qat_fixed_scale.item() == scale.item(), (
                f"calibration fixed scale must be equal to qat scale, "
                f"but get {scale.item()} in calibration and "
                f"{qat_fixed_scale.item()} in qat"
            )
        else:
            m.activation_post_process.activation_post_process = (
                fixed_scale_observer
            )
        if m.activation_post_process.observer_enabled.item() == 0:
            assert scale.item() == setted_scale.item()
    m.activation_post_process.set_qparams(scale, zero_point)
    if m.activation_post_process.observer_enabled.item() == 0:
        with torch.no_grad():
            m.activation_post_process.scale.copy_(setted_scale)


def _calculate_weight_qparams(m):
    from ..nn.qat import ConvTranspose2d

    if hasattr(m, "weight_fake_quant"):
        fake_quant_enabled = m.weight_fake_quant.fake_quant_enabled.item()
        observer_enabled = m.weight_fake_quant.observer_enabled.item()
        m.weight_fake_quant = m.qconfig.weight(
            channel_len=m.weight_fake_quant.channel_len
        ).to(m.weight.device)
        if hasattr(m, "_get_weight_for_fake_quant"):
            weight_for_fake_quant = m._get_weight_for_fake_quant()
        else:
            weight_for_fake_quant = m.weight
        m.weight_fake_quant.activation_post_process(weight_for_fake_quant)
        if fake_quant_enabled != 1:
            m.weight_fake_quant.disable_fake_quant()
        if observer_enabled != 1:
            m.weight_fake_quant.disable_observer()
        (
            scale,
            zero_point,
        ) = m.weight_fake_quant.calculate_qparams()
        m.weight_fake_quant.set_qparams(scale, zero_point)


def replace_fake_quantize(model):
    from .fake_quantize import CalibFakeQuantize

    for m in model.modules():
        if hasattr(m, "activation_post_process"):
            if isinstance(m.activation_post_process, CalibFakeQuantize):
                if m.qconfig.activation is not None:
                    _calculate_activation_qparams(m)
                else:
                    m.activation_post_process = None
                if m.qconfig.weight is not None:
                    _calculate_weight_qparams(m)
                else:
                    m.weight_fake_quant = None
    return model


def check_qat_qconfig(model):
    from .fake_quantize import CalibFakeQuantize

    for m in model.modules():
        if hasattr(m, "qconfig"):
            if m.qconfig is None:
                continue
            if m.qconfig.activation is None:
                continue
            fake_quant = m.qconfig.activation.p.func
            while hasattr(fake_quant, "p") and type(fake_quant.p) == partial:
                fake_quant = fake_quant.p.func
            if not fake_quant == CalibFakeQuantize:
                raise AttributeError(
                    "can not set qat qconfig before calibration qconfig"
                )


def _propagate_qconfig_helper(
    module, qconfig_dict, allow_list=None, qconfig_parent=None, prefix=""
):
    r"""This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to
                     quantization configuration
        allow_list: list of quantizable modules
        qconfig_parent: quantization config of parent module, we will fallback
                       to this config when there is no specified config for
                       current module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict

    Return:
        None, module is modified inplace with qconfig attached
    """
    # TODO: Add test
    if allow_list is None:
        allow_list = get_qconfig_propagation_list()

    module_qconfig = qconfig_dict.get(type(module), qconfig_parent)
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    module_qconfig = getattr(module, "qconfig", module_qconfig)
    # module can implement this method to modify qconfig of its children
    # when convert from calibratin to qat
    if hasattr(module, "propagate_qconfig"):
        module.propagate_qconfig(module_qconfig)

    module.qconfig = module_qconfig
    for name, child in module.named_children():
        module_prefix = prefix + "." + name if prefix else name
        _propagate_qconfig_helper(
            child, qconfig_dict, allow_list, module_qconfig, module_prefix
        )


def propagate_qconfig_(module, qconfig_dict=None, allow_list=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}
    _propagate_qconfig_helper(module, qconfig_dict, allow_list)


def prepare_calibration(model, inplace=False):
    """Prepare the model for calibration.

    Args:
        model: Float model with fused ops
        inplace: carry out model transformations in-place or not. Defaults to
        False.
    """
    check_qat_qconfig(model)
    model = prepare_qat(model, inplace=inplace)
    remove_qconfig(model)
    return model


def _is_in_out_same_scale_node(node, modules):
    # call_function and not floatfunctional call_method node, return true
    if node.op == "call_function":
        return True
    elif node.op == "call_method":
        from changan_plugin_pytorch.nn.quantized import FloatFunctional

        return not isinstance(modules[node.args[0].target], FloatFunctional)
    elif node.op == "call_module":
        m = modules[node.target]
        if isinstance(m, QuantStub):
            return False
        if (
            hasattr(m, "activation_post_process")
            and m.activation_post_process is not None
        ):
            return m.activation_post_process.observer_enabled.item() == 0
        else:
            return True
    else:
        assert (
            False
        ), "Should not encounter 'placeholder', 'get_attr' and 'output' node"


def _replace_inputs_scale(node_inputs_list, scale, modules):
    for m in node_inputs_list:
        # Process input args of cap.
        # 1) call_function or not 'FloatFunction' call_method node or input
        #    scale = output scale Module, process their inputs recursively
        # 2) otherwise, directly replace current node scale
        scale_setted_node = m
        if _is_in_out_same_scale_node(m, modules):
            # check and find the first input_scale != output_scale node
            if type(m.args[0]) == immutable_list:
                m_inputs_list = m.args[0]
            else:
                m_inputs_list = [m.args[0]]
            _replace_inputs_scale(m_inputs_list, scale, modules)
        else:
            if m.op == "call_method":
                from changan_plugin_pytorch.nn.quantized import FloatFunctional

                # Must be FloatFunctional method here now
                scale_setted_node = m.args[0]
                assert isinstance(
                    modules[scale_setted_node.target], FloatFunctional
                ), "Must be FloatFunctional here!"

            mod = modules[scale_setted_node.target]
            if (
                isinstance(mod, QuantStub) and mod.scale is not None
            ) or isinstance(mod.activation_post_process, FixedScaleObserver):
                assert False, (
                    "The scale of all inputs of 'cap' will be "
                    + "replaced with cap output scale by torch.fx. "
                    + "Please check your set scale!"
                )

            warnings.warn(
                "Observer of {} will be disabled and its scale will ".format(
                    mod.__class__.__name__
                )
                + "be replaced with cap output scale."
            )
            mod.activation_post_process.disable_observer()
            mod.activation_post_process.set_qparams(scale)


def pass_cat_same_scale(gm):
    """
    Replace cap input scale with cap result scale
    """
    modules = dict(gm.named_modules())
    for node in reversed(gm.graph.nodes):
        if node.op == "call_method" and node.target == "cap":
            # assume users always write FloatFunctional.cap
            # 'cap' node args is
            # (cat_op(actually is FloatFunction), [qfloat1, qfloat2, ...], dim)
            float_functional = modules[node.args[0].target]
            scale = float_functional.activation_post_process.scale
            _replace_inputs_scale(node.args[1], scale, modules)

    gm.recompile()
    return gm


def model_preprocess(model):
    """
    Preprocess model for some special purpose.
    Current only support unify cap input and output scale on Bernoulli by fx

    Args:
        model: the model to be preprocess

    Return:
        The GraphModel after preprocess

    """
    wrap_qat_modules_for_fx()
    tracer = CustomTracer()
    g = tracer.trace(model)
    gm = fx.GraphModule(model, g)
    return pass_cat_same_scale(gm)


def prepare_qat(
    model: torch.nn.Module,
    mapping: Optional[Dict[torch.nn.Module, torch.nn.Module]] = None,
    inplace: bool = False,
    optimize_graph: bool = False,
    hybrid: bool = False,
):
    r"""
    Prepares a copy of the model for quantization-aware training and
    converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
        optimize_graph: whether to do some process on origin model for special
                        purpose. Currently only support using torch.fx to fix
                        cap input scale(only used on Bernoulli)
        hybrid: whether to generate a hybrid model that some intermediate
                operation is computed in float. There are some constraints for
                this functionality now:
                1. The hybrid model cannot pass check_model and cannot be
                   compiled.
                2. Some quantized operation cannot directly accept input from
                   float operation, user need to manually insert QuantStub.
    """
    march = get_march()

    assert march is not None, "you must set march before invoking prepare_qat"
    assert isinstance(inplace, bool), "param 'inplace' must be bool type"

    qtensor_allow_float_operation(hybrid)

    if mapping is None:
        mapping = get_qat_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)

    propagate_qconfig_(
        model, qconfig_dict=None, allow_list=get_qconfig_propagation_list()
    )
    if after_calibration(model):
        replace_fake_quantize(model)
        # check and process 'cap' op in model on Bernoulli
        # 'model' is modified inplace
        if march == March.BERNOULLI and optimize_graph:
            gm = model_preprocess(model)
    else:
        convert(model, mapping=mapping, inplace=True, remove_qconfig=False)
    return model


def convert(module, mapping=None, inplace=False, remove_qconfig=True):
    r"""Converts submodules in input module to a different module according
    to `mapping` by calling `from_float` method on the target module class.
    And remove qconfig at the end if remove_qconfig is set to True.

    Args:
        module: input module
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated

    """
    march = get_march()

    assert march is not None, "you must set march before invoking convert"
    assert isinstance(inplace, bool), "argument 'inplace' must be of bool type"
    assert isinstance(
        remove_qconfig, bool
    ), "argument 'remove_qconfig' must be of bool type"

    if not inplace:
        module = copy.deepcopy(module)
    swapped_modules = {}
    with torch.no_grad():
        # disable autograd for all buffer copies
        _convert(
            module, mapping, inplace=True, swapped_modules=swapped_modules
        )
    if remove_qconfig:
        _remove_qconfig(module)
    return module


def _is_swappable_module(module, mapping):
    # find leaf module
    # TODO
    pass


def swap_module(mod, mapping, swapped_modules):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.
    copy from torch.quantization.quantize.swap_module, but judge

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module
        swapped_modules: a dictionary that maps from source module to swapped
                         module

    Return:
        The corresponding quantized module of `mod`
    """
    from torch.quantization import DeQuantStub

    new_mod = mod
    if mod in swapped_modules:
        new_mod = swapped_modules[mod]
    elif (
        hasattr(mod, "qconfig")
        and mod.qconfig is not None
        or isinstance(mod, DeQuantStub)
    ):
        swapped = False
        if type(mod) in mapping:
            from ..nn import Identity

            if (
                type(mod) in [torch.nn.Identity, Identity]
                or mapping[type(mod)] == Identity
            ):
                new_mod = mapping[type(mod)]()
            else:
                with torch.no_grad():
                    new_mod = mapping[type(mod)].from_float(mod)
            swapped_modules[mod] = new_mod
            swapped = True

        if swapped:
            # Preserve module's pre forward hooks. They'll be called on
            # quantized input
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            # Preserve module's post forward hooks
            # After convert they'll work with quantized output
            for hook_fn in mod._forward_hooks.values():
                new_mod.register_forward_hook(hook_fn)

            # respect device affinity when swapping modules
            devices = get_unique_devices_(mod)
            assert len(devices) <= 1, (
                "swap_module only works with cpu or single-device CUDA"
                " modules, but got devices {}".format(devices)
            )
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                new_mod.to(device)
    return new_mod


def _convert(module, mapping=None, inplace=False, swapped_modules={}):
    r"""Converts submodules in input module to a different module according
    to `mapping` by calling `from_float` method on the target module class

    Args:
        module: input module
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated
        swapped_modules: a dictionary that maps from source module to swapped
                         module

    """
    if mapping is None:
        mapping = get_quantized_operator_mappings()
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    SWAPPABLE_MODULES = set(get_qat_module_mappings().keys()) | set(
        get_quantized_operator_mappings().keys()
    )

    for name, mod in module.named_children():
        # both swappable modules and observed custom modules are
        # swapped as one unit
        if type(mod) not in SWAPPABLE_MODULES:
            _convert(
                mod, mapping, inplace=True, swapped_modules=swapped_modules
            )
        # TODO: judge swappable
        reassign[name] = swap_module(mod, mapping, swapped_modules)

    for key, value in reassign.items():
        setattr(module, key, value)

    return module
