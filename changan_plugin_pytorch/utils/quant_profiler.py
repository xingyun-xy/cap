import copy
import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from changan_plugin_pytorch import nn as horizon_nn
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn.quantized import FloatFunctional
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.quantization import quantization_mappings
from tabulate import tabulate
from torch import Tensor, nn
from torch.nn.modules.utils import _pair
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision import ops as vision_nn

from .model_helper import swap_ff_with_horizonff

__all__ = [
    "profile_module_constraints",
    "profile_featuremap",
    "featuremap_similarity",
    "get_module_called_count",
    "get_raw_features",
    "check_unfused_operations",
]


def _is_qat_mod(mod):
    return hasattr(mod, "qconfig") and (mod.config is not None)


def _warn_op_exist_by_march(marches=None, explanation=""):
    if marches is not None and not isinstance(marches, Iterable):
        marches = tuple(marches)

    def _func(mod, input, output):
        if _is_qat_mod(mod):
            return

        march = get_march()
        if marches is None or march in marches:
            op_name = str(mod.__class__)
            warnings.warn(
                "Operator %s detected and will harm the numerical accuracy"
                % op_name
                + " "
                + explanation
            )

    return _func


def _check_interpolate_inout_size(mod, input, output):
    if _is_qat_mod(mod):
        return

    def _check_quant_has_error(value, shift):
        int_value = value * (2 ** shift)
        return int_value - math.floor(int_value) > 0

    march = get_march()

    input_shape = input[0].shape
    output_shape = output.shape

    shift = 8 if march == March.BERNOULLI2 else 16

    if _check_quant_has_error(
        output_shape[2] / input_shape[2], shift
    ) or _check_quant_has_error(output_shape[3] / input_shape[3], shift):
        op_name = str(mod.__class__)
        warnings.warn(
            "Interpolation error will increase with output size"
            + " because the step is quantized %s" % op_name
        )


MODULE_CONSTRAINTS = {
    nn.AvgPool2d: lambda mod, input, output: warnings.warn(
        "Too large kernel size of nn.AvgPool2d"
        + " will harm the numerical accuracy"
    )
    if (_pair(mod.kernel_size)[0] * _pair(mod.kernel_size)[1] > 9)
    else None,
    nn.ReLU: _warn_op_exist_by_march(
        None, "Please consider replace it with ReLU6"
    ),
    nn.Sigmoid: _warn_op_exist_by_march(
        March.BERNOULLI2,
        "Because the output characteristic are not quantization friendly",
    ),
    nn.Softmax: _warn_op_exist_by_march(
        March.BAYES,
        "Because this op is implemented by look up reciprocal table",
    ),
    nn.SiLU: _warn_op_exist_by_march(
        March.BAYES,
        "Because this op is implemented by look up reciprocal table",
    ),
    horizon_nn.interpolate.Interpolate: _check_interpolate_inout_size,
    nn.Upsample: _check_interpolate_inout_size,
    nn.UpsamplingBilinear2d: _check_interpolate_inout_size,
    vision_nn.RoIAlign: _warn_op_exist_by_march(
        None, "Because the interpolate step is quantized"
    ),
}


class FunctionalWrapper:
    def __init__(self, mod):
        super(FunctionalWrapper, self).__init__()
        self.op_name = str(mod.__class__)
        self.mod = mod

    def div(self, x, y):
        warnings.warn(
            "Operator %s detected and will harm the numerical accuracy"
            % (self.op_name + ".div")
        )
        return self.mod.div(x, y)

    def cap(self, x, dim=0):
        if isinstance(x[0], QTensor):
            scales = torch.cat([data.q_scale().clone() for data in x])
            max_scale = scales.max()
            min_scale = scales.min()
            if max_scale / min_scale > 2:
                warnings.warn(
                    "Input scales of %s varies too much"
                    % (self.op_name + ".cap")
                    + " and will harm the numerical accuracy"
                )
        return self.mod.cap(x, dim)

    def __getattr__(self, name):
        return getattr(self.mod, name)


def _as_tuple(inputs):
    # Special case for common case of passing a single Tensor
    if isinstance(inputs, (torch.Tensor, dict)):
        inputs = (inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif not isinstance(inputs, tuple):
        inputs = tuple(inputs)
    return inputs


def profile_module_constraints(model, example_inputs):
    """Profile module contraints."""
    from ..quantization import quantization_mappings

    # init module constraints
    qat_mapping = quantization_mappings.get_qat_module_mappings()
    qat_constraints = {}
    for k, v in MODULE_CONSTRAINTS.items():
        if k in qat_mapping:
            qat_constraints[qat_mapping[k]] = v
    MODULE_CONSTRAINTS.update(qat_constraints)

    # register forward hook
    model = copy.deepcopy(model)
    functional_modules = {}
    for name, m in model.named_modules():
        if type(m) in (
            nn.quantized.FloatFunctional,
            horizon_nn.quantized.FloatFunctional,
        ):
            functional_modules[name] = FunctionalWrapper(m)
        if type(m) in MODULE_CONSTRAINTS:
            m.register_forward_hook(MODULE_CONSTRAINTS[type(m)])

    for k, v in functional_modules.items():
        model._modules[k] = v

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split(".")
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)

    setattr(cur_mod, tokens[-1], module)


def get_raw_features(model, example_inputs):
    """
    Use hooks to get raw features to be profiled.

    Args:
        model(Module): can be float/fused/calibration/qat/quantized model
        example_inputs(Any): the input data feed to model

    Returns:
        output(List(dict)): A list of dict. Each dict contains:
            "module_name": (str) the module name in the model
            "attr": (str) the attr of module. Maybe input/output/weight/bias
                Multi-inputs will be suffixed by input-i(i>=0)
            "data": (Tensor) the featuremap
            "scale": (Tensor, None) the scale of the feature if it has.
            "ch_axis": (int) channel axis of per channel quantized data.
    """
    model = copy.deepcopy(model)

    result = []

    def _record_data(data, mod, name):
        if isinstance(data, QTensor):
            result.append(
                {
                    "module_name": mod._debug_name,
                    "attr": name,
                    "data": data.dequantize(),
                    "scale": data.scale,
                    "ch_axis": data.q_per_channel_axis(),
                }
            )
        elif isinstance(data, (tuple, list)):
            if len(data) == 1:
                _record_data(data[0], mod, f"{name}")
            else:
                for i, d in enumerate(data):
                    _record_data(d, mod, f"{name}-{i}")
        elif isinstance(data, Tensor):
            result.append(
                {
                    "module_name": mod._debug_name,
                    "attr": name,
                    "data": data,
                    "scale": None,
                    "ch_axis": 0 if name == "weight" else -1,
                }
            )

    def _pre_hook(module, input):
        _record_data(input, module, "input")
        for name, param in module.named_parameters():
            if "." not in name:
                _record_data(param, module, name)
        for name, buf in module.named_buffers():
            if "." not in name:
                _record_data(buf, module, name)

    def _hook(module, input, output):
        _record_data(output, module, "output")

    qat_mapping = quantization_mappings.get_qat_module_mappings()
    quantized_mapping = quantization_mappings.get_quantized_operator_mappings()
    module_list = (
        set(qat_mapping.keys())
        | set(qat_mapping.values())
        | set(quantized_mapping.values())
        | set(
            [
                torch.nn.ReLU6,
            ]
        )
    )
    # register forward hook
    for name, m in model.named_modules():
        if isinstance(m, tuple(module_list)):
            # torch FloatFunctional methods do not call hooks
            # so replace it in float model with changan FloatFunctional
            if isinstance(m, torch.nn.quantized.FloatFunctional):
                m = horizon_nn.quantized.FloatFunctional()
                _set_module(model, name, m)
            # do not insert hook in FloatFunctional.activation_post_process
            # in float model
            if name.endswith("activation_post_process") and isinstance(
                m, (torch.nn.Identity, horizon_nn.Identity)
            ):
                continue
            m._debug_name = name
            m.register_forward_pre_hook(_pre_hook)
            m.register_forward_hook(_hook)

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)

    return result


def profile_featuremap(
    featuremap,
    with_tensorboard=False,
    tensorboard_dir=None,
    print_per_channel_scale=False,
):
    """Profile featuremap value with log or tensorboard.
    Print min/max/mean/var/scale of each feature profiled by `get_raw_features`
    by default. If `with_tensorboard` set True, the histogram of each feature
    will be shown in tensorboard, which is useful to see the data distribution.

    If you want to get more info about features, you can define your custom
    profile functions to process the results of `get_raw_features`.

    Args:
        featuremap(list(dict)): raw featuremaps returned by `get_raw_features`
        with_tensorboard(bool): whether to use tensorboard. Default: False
        tensorboard_dir(str, None): tensorboard log file path. Default: None
        print_per_channel_scale(bool): whether to print per channel scales.
            Default: False
    """
    table = []
    per_channel_table = []
    for f in featuremap:
        fdata = f["data"].float()
        fmin, fmax, fmean, fvar = (
            fdata.min().item(),
            fdata.max().item(),
            fdata.mean().item(),
            fdata.var().item(),
        )
        fvar = "" if math.isnan(fvar) else fvar
        if f["scale"] is None:
            table.append(
                (f["module_name"], f["attr"], fmin, fmax, fmean, fvar, "")
            )
        elif f["scale"].numel() == 1:
            table.append(
                (
                    f["module_name"],
                    f["attr"],
                    fmin,
                    fmax,
                    fmean,
                    fvar,
                    f["scale"].item(),
                )
            )
        else:
            table.append(
                (
                    f["module_name"],
                    f["attr"],
                    fmin,
                    fmax,
                    fmean,
                    fvar,
                    "per channel scale",
                )
            )
            per_channel_table.append(
                (
                    f["module_name"],
                    f["attr"],
                    len(f["scale"]),
                    f["ch_axis"],
                    f["scale"].cpu(),
                )
            )
    print("\n")
    print(
        tabulate(
            table,
            headers=[
                "Module Name",
                "Input/Output/Attr",
                "Min",
                "Max",
                "Mean",
                "Var",
                "Scale",
            ],
            tablefmt="psql",
            floatfmt=".10f",
            numalign="left",
        )
    )

    if print_per_channel_scale:
        print(
            tabulate(
                per_channel_table,
                headers=[
                    "Module Name",
                    "Input/Output",
                    "Channel Len",
                    "Channel Axis",
                    "Scale",
                ],
                tablefmt="grid",
                numalign="left",
            )
        )

    if with_tensorboard:
        writer = SummaryWriter(log_dir=tensorboard_dir)
        for f in featuremap:
            # shown per channel quantized weight and features
            if f["ch_axis"] != -1:
                for i in range(f["data"].shape[f["ch_axis"]]):
                    writer.add_histogram(
                        f"{f['module_name']}:{f['attr']}",
                        # ch_axis = 0 or 1
                        f["data"][i] if f["ch_axis"] == 0 else f["data"][:, i],
                        i,
                    )
            # tensorboard histogram result is confused when only one number
            elif f["data"].numel() != 1:
                writer.add_histogram(
                    f"{f['module_name']}:{f['attr']}", f["data"]
                )
        writer.close()


def featuremap_similarity(
    model1,
    model2,
    inputs,
    similarity_func="Cosine",
    threshold=None,
):
    """
    Compute the similarity of feature maps. The input models can be float/
    fused/calibration/qat/quantized model.

    Args:
        model1(Module): can be float/fused/calibration/qat/quantized model
        model2(Module): can be float/fused/calibration/qat/quantized model
        inputs(Any): the input data feed to model
        similarity_func(str, Callable): similarity computation function.
                        Support "Cosine", "MSE", "L1", "KL" or any user-defined
                        Callable object. If it is a user-defined object, it
                        should return a scalar or tensor with only one number.
                        Otherwise the result shown may be unexpected.
                        Default: "Cosine"
        threshold(float, None): if similarity value exceeds or less than this
                        threshold, the featuremap name will be shown in red
                        color.If threshold is none, it will be set to different
                        values according to different similarity functions
                        Default: None
    """
    assert callable(similarity_func) or similarity_func in (
        "Cosine",
        "MSE",
        "L1",
        "KL",
    ), "Unsupport similarity computation function {}!".format(similarity_func)
    if similarity_func == "Cosine":
        func = torch.nn.CosineSimilarity(dim=0)
    elif similarity_func == "MSE":
        func = torch.nn.MSELoss()
    elif similarity_func == "L1":
        func = torch.nn.L1Loss()
    elif similarity_func == "KL":
        func = torch.nn.KLDivLoss()
    else:
        func = similarity_func

    if threshold is None:
        threshold = 0.0 if similarity_func == "Cosine" else 1.0

    # The closer the value of cosine similarity_func result gets to 1.0, the
    # two layers results are more similar. It reverses in other similarity func
    compare_func = (
        lambda x: x <= threshold
        if similarity_func == "Cosine"
        else x >= threshold
    )

    model1 = copy.deepcopy(model1)
    model2 = copy.deepcopy(model2)
    qat_mapping = quantization_mappings.get_qat_module_mappings()
    quantize_mappings = quantization_mappings.get_quantized_operator_mappings()
    leaf_module_list = (
        list(qat_mapping.keys())
        + list(qat_mapping.values())
        + list(quantize_mappings.values())
    )
    leaf_module_list.append(torch.nn.ReLU6)

    model1_mod2name = dict()
    model2_mod2name = dict()
    for name, mod in model1.named_modules():
        model1_mod2name[mod] = name
    for name, mod in model2.named_modules():
        model2_mod2name[mod] = name

    # find the modules to insert the hook and process different modules
    # for example:
    #   float module: softmax = nn.Softmax()
    #   qat/quantized module: softmax = SegmentLUTSoftmax(sub, exp, sum, reciprocal, mul) # noqa
    # we only insert hook in 'softmax' layer, not submodules in qat softmax
    # Note: if computing similarity of qat and quantized softmax modules, which
    #   are matched, hooks will be inserted normally in submodules.
    def _find_leaf_modules(module, leafset, mod2name):
        for name, m in module.named_children():
            if type(m) in tuple(leaf_module_list):
                leafset.add(mod2name[m])
            else:
                _find_leaf_modules(m, leafset, mod2name)
        return leafset

    def _has_submodule(module, target):
        if target == "":
            return True

        atoms = target.split(".")
        mod = module
        for item in atoms:
            if not hasattr(mod, item):
                return False
            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                return False
        return True

    model1_leafs = _find_leaf_modules(model1, set(), model1_mod2name)
    model2_leafs = _find_leaf_modules(model2, set(), model2_mod2name)
    leafs = model1_leafs | model2_leafs
    # delete nonexistent modules
    discard_leafs = set()
    for leaf in leafs:
        if not _has_submodule(model1, leaf) or not _has_submodule(
            model2, leaf
        ):
            discard_leafs.add(leaf)
    leafs = leafs - discard_leafs

    def _register_hook(module, _hook, leaf_module_set):
        for name, m in module.named_modules():
            if name in leaf_module_set:
                # torch FloatFunctional methods do not call hooks
                # so replace it in float model with changan FloatFunctional
                if isinstance(m, torch.nn.quantized.FloatFunctional):
                    m = horizon_nn.quantized.FloatFunctional()
                    _set_module(module, name, m)
                m._debug_name = name
                m._shared_times = 0
                m.register_forward_hook(_hook)

    # register hook for module1
    MODULE1_FMAP = OrderedDict()

    def _hook(module, input, output):
        if module._shared_times > 0:
            MODULE1_FMAP[
                module._debug_name + "(" + str(module._shared_times) + ")"
            ] = output
        else:
            MODULE1_FMAP[module._debug_name] = output
            module._shared_times += 1

    _register_hook(model1, _hook, leafs)

    # register hook for module2
    MODULE2_FMAP = OrderedDict()

    def _hook(module, input, output):
        if module._shared_times > 0:
            MODULE2_FMAP[
                module._debug_name + "(" + str(module._shared_times) + ")"
            ] = output
        else:
            MODULE2_FMAP[module._debug_name] = output
            module._shared_times += 1

    _register_hook(model2, _hook, leafs)

    inputs = _as_tuple(inputs)
    model1(*inputs)
    model2(*inputs)

    # process featuremap
    for m in (MODULE1_FMAP, MODULE2_FMAP):
        for k, v in m.items():
            m[k] = m[k].dequantize() if isinstance(m[k], QTensor) else m[k]

    from ..nn import Identity

    result = []
    for k, v in MODULE1_FMAP.items():
        if k not in MODULE2_FMAP:
            # should not run here!
            warnings.warn("key {} not found in MODULE2_FMAP".format(k))
            continue
        if similarity_func == "KL":
            # model1 is considered as target result
            # use fp64 to improve computation precision
            ret = func(
                F.log_softmax(MODULE2_FMAP[k].to(torch.float64), dim=1),
                F.softmax(v.to(torch.float64), dim=1),
            )
        elif similarity_func == "Cosine":
            ret = func(v.flatten(), MODULE2_FMAP[k].flatten()).item()
        else:
            ret = func(v, MODULE2_FMAP[k])
        is_model1_identity = isinstance(
            model1.get_submodule(
                k
                if k in model1_mod2name.values()
                else k[: -len(k.split("(")[-1]) - 1]
            ),
            (torch.nn.Identity, Identity),
        )
        is_model2_identity = isinstance(
            model2.get_submodule(
                k
                if k in model2_mod2name.values()
                else k[: -len(k.split("(")[-1]) - 1]
            ),
            (torch.nn.Identity, Identity),
        )
        if is_model1_identity and is_model2_identity:
            suffix = "(I vs I)"
        elif is_model1_identity or is_model2_identity:
            suffix = "(I)"
        else:
            suffix = ""
        k = k + suffix
        if compare_func(ret):
            k = "\033[31m" + k + "\033[0m"
            ret = "\033[31m{:.17f}\033[0m".format(ret)
        result.append((k, ret))

    print("\n{:-^{width}}".format("-", width=63))
    print("Note:")
    print("* Suffix '(I)' means this layer is Identity in one model")
    print("* Suffix '(I vs I)' means this layer is Identity in both models")
    print("* Suffix '(i)'(i >= 1) means this op is shared i times")
    print("{:-^{width}}".format("-", width=63))
    print(
        tabulate(
            result,
            headers=["Module Name", "Similarity"],
            tablefmt="psql",
            floatfmt=".17f",
            numalign="left",
        )
    )


def attach_qualified_name(model: torch.nn.Module) -> None:
    """Attach qualified name to all named modules"""
    for name, module in model.named_modules(remove_duplicate=False):
        if hasattr(module, "_qualified_name"):
            warnings.warn(
                "{} and {} refer to the same instance, "
                "we will use the former one as module name".format(
                    module._qualified_name, name
                ),
                UserWarning,
            )
        else:
            module._qualified_name = name


SUPPORTED_MODULE_CLASSES = (
    set(quantization_mappings.get_qat_module_mappings().keys())
    | set(quantization_mappings.get_qat_module_mappings().values())
    | set(quantization_mappings.get_quantized_operator_mappings().keys())
    | set(quantization_mappings.get_quantized_operator_mappings().values())
)


def is_leaf_module(module: torch.nn.Module) -> bool:
    """Check if a module is leaf"""
    if type(module) in SUPPORTED_MODULE_CLASSES:
        return True

    if module.__module__.startswith("torch.nn") and not isinstance(
        module, torch.nn.Sequential
    ):
        # unsupported float module should be treated as leaf
        return True

    if all(False for _ in module.named_children()):
        # for user defined module which has no children
        return True

    return False


def register_hook_on_leaf(
    model: torch.nn.Module,
    forward_hook: callable = None,
    forward_pre_hook: callable = None,
    check_leaf_module: callable = None,
    prefix: str = "",
) -> Dict[str, Tuple[RemovableHandle, RemovableHandle]]:
    """
    Register forward_hook and forward_pre_hook on all leaf modules in a model.

    Args:
        model (torch.nn.Module): The input model.
        forward_hook (callable, optional): forward_hook to register.
            Defaults to None.
        forward_pre_hook (callable, optional): forward_pre_hook to register.
            Defaults to None.
        check_leaf_module (callable, optional): A function to check if
            a module is leaf. Pass None to use pre-defined `is_leaf_module`.
            Defaults to None.
        prefix (str, optional): The name of root module, only for internal use.
            Defaults to "".

    Returns:
        Dict[str, Tuple[RemovableHandle, RemovableHandle]]:
            A mapping from module's qualified name to the handler of
            registered hooks.
    """
    if check_leaf_module is None:
        check_leaf_module = is_leaf_module

    handler_dict = {}

    for name, module in model.named_children():
        if check_leaf_module(module):
            handler = [None, None]
            if forward_hook is not None:
                handler[0] = module.register_forward_hook(forward_hook)
            if forward_pre_hook is not None:
                handler[1] = module.register_forward_pre_hook(forward_pre_hook)
            handler_dict[prefix + name] = handler
        else:
            handler_dict.update(
                register_hook_on_leaf(
                    module,
                    forward_hook,
                    forward_pre_hook,
                    check_leaf_module,
                    prefix + name + ".",
                )
            )

    return handler_dict


def get_module_called_count(
    model: torch.nn.Module,
    example_inputs,
    check_leaf_module: callable = None,
    print_tabulate: bool = True,
) -> Dict[str, int]:
    """
    Count called times for all leaf modules in a model.

    Args:
        model (torch.nn.Module): The input model.
        example_inputs (Any[Tensor]): The input data feed to model.
        check_leaf_module (callable, optional): A function to check if
            a module is leaf. Pass None to use pre-defined `is_leaf_module`.
            Defaults to None.
        print_tabulate (bool, optional): Whether print the result as tabulate.
            Defaults to True.

    Returns:
        Dict[str, int]:
            The qualified name and called times of each leaf module.
    """
    if check_leaf_module is None:
        check_leaf_module = is_leaf_module

    model = copy.deepcopy(model)

    swap_ff_with_horizonff(model)
    attach_qualified_name(model)

    module_refcount = {}

    def _count_call_hook(module, input, output):
        module_refcount[module._qualified_name] += 1

    handler_dict = register_hook_on_leaf(
        model, _count_call_hook, check_leaf_module=check_leaf_module
    )

    for name in handler_dict:
        module_refcount[name] = 0

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)
    del model

    if print_tabulate:
        print(
            tabulate(module_refcount.items(), headers=["name", "called times"])
        )

    return module_refcount


def check_unfused_operations(
    model: torch.nn.Module, example_inputs, print_tabulate=True
):
    """
    Check unfused modules in a model.
    NOTE: This function is only capable to check unfused modules. For the
          correctness of fusion, please use `featuremap_similarity`
          to compare the feature between fused and unfused model.

    Args:
        model (torch.nn.Module):  The input model.
        example_inputs (Any[Tensor]): The input data feed to model.
        print_tabulate (bool, optional): Whether print the result as tabulate.
            Defaults to True.

    Returns:
        List[List[str]]:
            The qualified name of modules that can be fused.
    """
    model = copy.deepcopy(model)

    swap_ff_with_horizonff(model)
    attach_qualified_name(model)

    output_to_module_mapping = {}
    virtual_input_node = nn.Identity()
    virtual_input_node._qualified_name = "placeholder"
    virtual_input_node._output_to = []

    def _add_virtual_input_node(inputs):
        if isinstance(inputs, (Tensor, QTensor)):
            output_to_module_mapping[inputs] = virtual_input_node
        # in CAP config, dict is ConfigDict, a subclass of dict
        elif issubclass(type(inputs), dict):
            for value in list(inputs.values()):
                _add_virtual_input_node(value)
        elif type(inputs) in (list, tuple):
            for i in inputs:
                _add_virtual_input_node(i)

    _add_virtual_input_node(example_inputs)

    def _record_output_hook(module, input, output):
        if isinstance(output, (list, tuple)):
            for x in output:
                _record_output_hook(module, input, x)
        elif isinstance(output, (Tensor, QTensor)):
            output_to_module_mapping[output] = module

    def _make_graph_hook(module, input):
        if not hasattr(module, "_input_from"):
            module._input_from = []
        if not hasattr(module, "_output_to"):
            module._output_to = []

        if isinstance(input, (list, tuple)):
            for x in input:
                _make_graph_hook(module, x)
        elif isinstance(input, (Tensor, QTensor)):
            input_from = output_to_module_mapping[input]
            if input_from not in module._input_from:
                module._input_from.append(input_from)
            if module not in input_from._output_to:
                input_from._output_to.append(module)

    register_hook_on_leaf(model, _record_output_hook, _make_graph_hook)

    example_inputs = _as_tuple(example_inputs)
    model(*example_inputs)

    def match_node(module, expected_type):
        if hasattr(module, "_matched"):
            return False
        if not type(module) == expected_type:
            return False
        if expected_type is FloatFunctional:
            return module._last_called_method_name == "add"
        return True

    def match_pattern(root: torch.nn.Module, pattern):
        if match_node(root, pattern[0]):
            if len(pattern) == 1:
                root._matched = True
                return [[root]]
            else:
                tile = []
                for next_node in root._output_to:
                    tile += match_pattern(next_node, pattern[1:])
                for matched_seq in tile:
                    root._matched = True
                    matched_seq.insert(0, root)
                return tile
        else:
            return []

    def get_unmatched_next(root: torch.nn.Module):
        next_nodes = set()
        for node in root._output_to:
            if hasattr(node, "_matched"):
                next_nodes |= get_unmatched_next(node)
            else:
                next_nodes.add(node)

        return next_nodes

    def search_pattern(root: torch.nn.Module, patterns):
        current_roots = [root]
        ret = []

        while len(current_roots) > 0:
            next_roots = set()
            for node in current_roots:
                if not hasattr(node, "_matched"):
                    matched_seqs = []
                    for pattern in patterns:
                        matched_seqs = match_pattern(node, pattern)
                        if len(matched_seqs) > 0:
                            ret += matched_seqs
                            break

                next_roots |= get_unmatched_next(node)
            if current_roots == next_roots:
                warnings.warn(
                    "There are circles in the graph, "
                    "related nodes are {}".format(
                        [m._qualified_name for m in current_roots]
                    ),
                    UserWarning,
                )
                break
            current_roots = next_roots

        return ret

    from changan_plugin_pytorch.quantization.fuse_modules import (
        get_op_list_to_fuser_mapping,
    )

    fuse_patterns = list(get_op_list_to_fuser_mapping().keys())
    fuse_patterns.sort(key=len, reverse=True)

    matched_seqs = search_pattern(virtual_input_node, fuse_patterns)
    module_to_fuse = []
    for matched_seq in matched_seqs:
        valid = True
        shared_conv = False
        if len(matched_seq[0]._output_to) > 1:
            shared_conv = True
        for m in matched_seq[1:]:
            if isinstance(m, FloatFunctional):
                if len(m._input_from) > 2 or len(m._output_to) > 1:
                    valid = False
                    break
            else:
                if len(m._input_from) > 1 or len(m._output_to) > 1:
                    valid = False
                    break

        if valid:
            module_to_fuse.append(
                [
                    (
                        module._qualified_name
                        + ("(shared)" if i == 0 and shared_conv else ""),
                        type(module),
                    )
                    for i, module in enumerate(matched_seq)
                ]
            )

    if print_tabulate:
        if len(module_to_fuse) == 0:
            print("Do not find any fusable modules")
        else:
            print("Fusable modules are listed below\n")
            for item in module_to_fuse:
                print(
                    tabulate(
                        item + [["", ""]],
                        headers=["name", "type"],
                    )
                )

    del model

    return module_to_fuse
