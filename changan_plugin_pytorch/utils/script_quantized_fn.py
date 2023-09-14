r"""Extended wrapper of torch.jit.script_if_tracing
This file defines a wrapper behaves like torch.jit.script_if_tracing, and
does some process for exporting quantized model to ONNX. Directly using
torch.jit.script causes unsupported operations in ONNX.
"""
import functools
import inspect
from typing import List, Tuple

import torch
from torch import Tensor
import itertools


# flatten lists in functions outputs
def _filter_encode(origin_list_out):
    return tuple(itertools.chain.from_iterable(origin_list_out))


def _dppv1_encode(origin_list_out):
    return tuple(itertools.chain.from_iterable(origin_list_out))


def _dpp_encode(origin_list_out):
    return tuple(itertools.chain.from_iterable(origin_list_out))


def _anchor_generator_encode(origin_list_out):
    return tuple(origin_list_out)


def _multi_scale_roi_align_encode(origin_list_out):
    return tuple(itertools.chain.from_iterable(origin_list_out))


# resume original outputs from flattened outputs
# filter output: List[List[Tensor]]
def _filter_decode(packed_out, *args):
    N = args[0].shape[0]
    input_num = len(args[-4])  # len of input_dtype_list

    ret_list: List[List[Tensor]] = []
    for i in range(0, len(packed_out), N):
        per_image_ret = packed_out[i : i + 3 + input_num]
        ret_list.append(per_image_ret)
    return ret_list


# detection_post_process_v1 output: List[Tuple[Tensor, Tensor, Tensor]]
def _dppv1_decode(packed_out, *args):
    ret_list: List[Tuple[Tensor, Tensor, Tensor]] = []
    for i in range(0, len(packed_out), 3):
        ret_list.append(packed_out[i : i + 3])
    return ret_list


def _dpp_decode(packed_out, *args):
    ret_list: List[Tuple[Tensor]] = []
    # args[0] is 'self'
    per_len = args[1].shape[0]
    for i in range(0, len(packed_out), per_len):
        ret_list.append(packed_out[i : i + per_len])
    return tuple(ret_list)


def _anchor_generator_decode(packed_out, *args):
    return tuple(packed_out)


def _multi_scale_roi_align_decode(packed_out, *args):
    return tuple(packed_out)

def script_quantized_fn(fn):
    """
    This function can be used as a decorator to wrap a quantized op function.

    if is_in_onnx_export():
        define and apply a corresponding autograd function to export to ONNX
        1) Flatten List[Tensor] in function inputs(because autograd function
           do not support List[Tensor] input), then resume original input
           format in autograd function forward to call quantized function.
        2) Flatten List output, otherwise ONNX will record the operations in
           the function to construct the List output so that the function can
           not be exported as one node in ONNX. The original outputs will be
           resumed after applying autograd function.
    else:
        behaves like torch.jit.script_if_tracing
    """

    # functions with list outputs must be special processed
    list_output_fns_mapping = {
        "filter": (_filter_encode, _filter_decode),
        "detection_post_process_v1": (_dppv1_encode, _dppv1_decode),
        "AnchorGenerator": (
            _anchor_generator_encode,
            _anchor_generator_decode,
        ),
        # "MultiScaleRoIAlign": (
        #     _multi_scale_roi_align_encode,
        #     _multi_scale_roi_align_decode,
        # ),
        "DetectionPostProcessV1": (_dppv1_encode, _dppv1_decode),
        "DetectionPostProcess": (_dpp_encode, _dpp_decode),
    }

    (
        arg_names,
        varargs,
        varkw,
        defaults,
        kwonlyargs,
        kwonlydefaults,
        annotations,
    ) = inspect.getfullargspec(fn)

    # support float module.forward
    fn_or_module_name = (
        fn.__qualname__.split(".")[0]
        if fn.__name__ == "forward"
        else fn.__name__
    )

    # find List[Tensor] inputs index
    list_of_tensor_arg_idx = []
    for name, anno in annotations.items():
        if anno == List[torch.Tensor] and name != "return":
            list_of_tensor_arg_idx.append(arg_names.index(name))

    if list_of_tensor_arg_idx:

        def construct_args(per_list_lens, args):
            """
            This function resumes original inputs format from flattened args
            """
            args = list(args)
            for i, idx in enumerate(list_of_tensor_arg_idx):
                tensor_list = []
                for _ in range(per_list_lens[i]):
                    tensor_list.append(args.pop(idx))
                args.insert(idx, tensor_list)

            return args

        class Vslz(torch.autograd.Function):
            @staticmethod
            def forward(ctx, name, per_list_lens, *args):
                # resume original inputs to call original quantized function
                args = construct_args(per_list_lens, args)
                out = fn(*args)
                # flatten list outputs to avoid ONNX recording operations
                # in the quantized function
                if fn_or_module_name in list_output_fns_mapping.keys():
                    out = list_output_fns_mapping[fn_or_module_name][0](out)
                return out

    else:

        class Vslz(torch.autograd.Function):
            @staticmethod
            def forward(ctx, name, per_list_lens, *args):
                out = fn(*args)
                if fn_or_module_name in list_output_fns_mapping.keys():
                    out = list_output_fns_mapping[fn_or_module_name][0](out)
                return out

    def apply_function(*args, **kwargs):
        args_num = len(args)
        total_args = []  # flattened inputs
        per_list_lens = []  # per list length of lists in input args

        # flatten list in the inputs
        for i, arg_name in enumerate(arg_names):
            if i < args_num:
                current_arg = args[i]
            elif arg_name in kwargs:
                current_arg = kwargs[arg_name]
            else:
                current_arg = defaults[i - len(arg_names)]

            if i in list_of_tensor_arg_idx:
                total_args.extend(current_arg)
                per_list_lens.append(len(current_arg))
            else:
                total_args.append(current_arg)

        out = Vslz.apply(fn.__qualname__, per_list_lens, *total_args)

        # resume the original quantized function output
        if fn_or_module_name in list_output_fns_mapping.keys():
            out = list_output_fns_mapping[fn_or_module_name][1](
                out, *total_args
            )
        return out

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.onnx.is_in_onnx_export():
            return apply_function(*args, **kwargs)
        # do not script if fn is module.forward or segment_lut methods
        elif not torch.jit.is_tracing() or fn.__name__ in (
            "forward",
            "_init_single_table_params",
            "_init_multi_table_params",
        ):
            return fn(*args, **kwargs)

        compiled_fn = torch.jit.script(fn)
        return compiled_fn(*args, **kwargs)

    wrapper.list_of_tensor_arg_idx = list_of_tensor_arg_idx
    wrapper.__original_fn = fn
    wrapper.__script_if_tracing_wrapper = True

    return wrapper
