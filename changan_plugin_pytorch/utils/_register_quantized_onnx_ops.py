r"""Register quantized ops for ONNX.
This file registers a custom symbolic function
for exporting quantized models to ONNX.
"""
import inspect
import warnings
from numbers import Integral, Real

import changan_plugin_pytorch
import torch
from torch.onnx import register_custom_op_symbolic
from cap.utils.apply_func import flatten


def symbolic_quantized_op(
    g: torch._C.Graph, n: torch._C.Node, *args, **kwargs
):
    r"""
    Register quantized ops for ONNX. This function will flatten List[Tensor]
    inputs and outputs to single Tensor because autograd function do not
    support List[Tensor] now.

    Note: Must be used with `script_quantized_fn` defined in
          script_quantized_fn.py
    """

    # args format:
    # (
    #     function name,
    #     tuple of per list lengths of 'List[Tensor]' args in input args,
    #     input args with flatten list inputs,
    # )
    # example:
    #   origin args: func(Tensor, int, [Tensor1, Tensor2], int)
    #   processed args: (func_name, 2, Tensor, int, Tensor1, Tensor2, int)
    fn_name, per_list_lens, *args = args

    if not isinstance(fn_name, str):
        return None

    module = None
    if "." in fn_name:
        # float op forward or segment_lut methods
        module_name, func_name = fn_name.split(".")
        # module is 'self' arg in forward function
        module, *args = args
        if func_name == "forward":
            func = getattr(
                changan_plugin_pytorch.nn, module_name, None
            ).forward
            fn_name = module_name  # use module_name in onnx
        elif func_name in (
            "_init_single_table_params",
            "_init_multi_table_params",
        ):
            func = getattr(
                changan_plugin_pytorch.nn.quantized.SegmentLUT, func_name, None
            )
        else:
            raise ValueError("Unknown qualname {}".format(fn_name))
    else:
        func = getattr(
            changan_plugin_pytorch.nn.quantized.functional, fn_name, None
        )
    if func is None:
        return None

    (
        arg_names,
        varargs,
        varkw,
        defaults,
        kwonlyargs,
        kwonlydefaults,
        annotations,
    ) = inspect.getfullargspec(func.__original_fn)
    # do not show 'self' arg in onnx
    arg_names = arg_names[1:] if module is not None else arg_names

    arg_idx_mapping = list(range(len(arg_names)))

    # list_of_tensor_arg_idx:
    #   A list of List[Tensor] args indexes in func.
    #   If no List[Tensor] args, will be []
    if func.list_of_tensor_arg_idx:
        # update idx to find args name in origin args
        list_arg_len_map = zip(
            reversed(func.list_of_tensor_arg_idx), reversed(per_list_lens)
        )
        for idx, list_len in list_arg_len_map:
            for _ in range(list_len - 1):
                arg_idx_mapping.insert(idx, idx)

    code = 'g.op("changan::{}", '.format(fn_name)

    # put Tensor args in the front
    for i, arg in enumerate(args):
        if isinstance(arg, torch._C.Value):
            code += "args[{}], ".format(i)

    type_to_reg_mapping = {
        Real: "f",
        Integral: "i",
        bool: "i",
        str: "s",
        torch.Tensor: "t",
    }

    def get_type(arg):
        if isinstance(arg, Integral):
            return Integral
        elif isinstance(arg, Real):
            return Real
        elif isinstance(arg, str):  # for QuantDtype
            return str
        else:
            return type(arg)

    # process not Tensor args
    for i, arg in enumerate(args):
        reg_annt = type_to_reg_mapping.get(get_type(arg), None)
        # process list and tuple of (int, float, bool, s)
        if reg_annt is None:
            if isinstance(arg, (list, tuple)):
                reg_annt = type_to_reg_mapping.get(get_type(arg[0]), None)

        # Not support type arg(and arg is not None) will be converted to str
        if (
            arg is not None
            and reg_annt is None
            and not isinstance(arg, torch._C.Value)
        ):
            warnings.warn(
                "FUNCTION '{}' ARG '{}' type is {}, ".format(
                    fn_name, arg_names[arg_idx_mapping[i]], type(arg)
                )
                + "which is not support in ONNX, will be converted to 'str'."
            )
            reg_annt = "s"
            args[i] = str(args[i])

        if reg_annt is not None:
            code += "{}_{}=args[{}], ".format(
                arg_names[arg_idx_mapping[i]], reg_annt, i
            )

    output_nodes = list(n.outputs())

    code = code[:-2] + ", outputs={})".format(len(output_nodes))

    if fn_name == 'AnchorGenerator':
        code = code[:-1] + ", feat_strides_i={}".format(
            module.feat_strides, module.anchor_wh_groups
        )
        code += ", image_hw_i={}".format(
            module.image_hw
        )
        for stride, anchor_wh_groups in zip(module.feat_strides, module.anchor_wh_groups):
            code += ", anchor_wh_groups_{}_i={}".format(stride, flatten(anchor_wh_groups)[0])
        code += ", legacy_bbox_i={})".format(module.legacy_bbox)
    elif fn_name == 'DetectionPostProcessV1':
        code = code[:-1] + ", num_classes_i={}".format(
            module.num_classes
        )
        code += ", class_offsets_i={}".format(
            module.class_offsets
        )
        code += ", use_clippings_i={}".format(
            module.use_clippings
        )
        code += ", image_hw_i={}".format(
            module.image_size
        )
        code += ", nms_iou_threshold_f={}".format(
            module.nms_threshold
        )
        code += ", nms_margin_f={}".format(
            module.nms_margin
        )
        code += ", box_filter_threshold_f={}".format(
            module.box_filter_threshold
        )
        code += ", pre_nms_top_k_i={}".format(
            module.pre_nms_top_k
        )
        code += ", post_nms_top_k_i={}".format(
            module.post_nms_top_k
        )
        # code += ", nms_padding_mode_s={}".format(
        #     str(module.nms_padding_mode)
        # )
        code += ", bbox_min_hw_i={})".format(
            module.bbox_min_hw
        )
    elif fn_name == 'MultiScaleRoIAlign':
        code = code[:-1] + ", output_size_i={}".format(
            module.output_size
        )
        code += ", image_hw_i={}".format(
            module.image_hw
        )
        code += ", feature_strides_i={}".format(
            module.feature_strides
        )
        code += ", canonical_level_i={}".format(
            module.canonical_level
        )
        code += ", aligned_i={})".format(
            module.aligned
        )

    ret = eval(code)

    if isinstance(ret, (list, tuple)):
        list_ret = ret
    else:
        list_ret = [ret]
    for r, node in zip(list_ret, output_nodes):
        r.setType(node.type())

    return ret


register_custom_op_symbolic("::prim_PythonOp", symbolic_quantized_op, 1)
