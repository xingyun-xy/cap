import torch.onnx
from torch.onnx import OperatorExportTypes
import torch._C as _C
import warnings
from . import _register_onnx_ops
from . import _register_quantized_onnx_ops
import os
import sys
import contextlib

TrainingMode = _C._onnx.TrainingMode


__all__ = ["export_to_onnx", "export_quantized_onnx"]


# torch 1.10.2 add some logic in onnx shape inference and use std::cerr
# print warnings in custom registered ops.
# We redirect stderr to null to avoid warnings in each custom op,
# do torch.onnx.export and then redirect stderr back.
@contextlib.contextmanager
def _redirect_stderr():
    # Note: Directly use sys.stderr.fileno() cause 'Tee' error in CI/CD
    # stderr_fd = sys.stderr.fileno()
    stderr_fd = 2
    fd = os.open("/dev/null", os.O_WRONLY)
    dup_stderr_fd = os.dup(stderr_fd)
    try:
        yield os.dup2(fd, stderr_fd)
    finally:
        os.dup2(dup_stderr_fd, stderr_fd)
        os.close(fd)
        os.close(dup_stderr_fd)


def export_to_onnx(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    opset_version=11,
    do_constant_folding=True,
    example_outputs=None,
    dynamic_axes=None,
    enable_onnx_checker=False,
):
    r"""
    Export a (float or qat)model into ONNX format.

    Args:
        model (torch.nn.Module, torch.jit.ScriptModule , \
            or torch.jit.ScriptFunction): the model to be exported.
        args (tuple or torch.Tensor):

            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS::

                args = (x, y, z)

            The tuple should contain model inputs such that ``model(*args)`` \
            is a valid invocation of the model. Any non-Tensor arguments will \
            be hard-coded into the exported model; any Tensor arguments will \
            become inputs of the exported model, in the order they occur in \
            the tuple.

            2. A TENSOR::

                args = torch.Tensor([1])

            This is equivalent to a 1-ary tuple of that Tensor.

            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED \
            ARGUMENTS::

                args = (x,
                        {'y': input_y,
                         'z': input_z})

            All but the last element of the tuple will be passed as non-keyword \
            arguments, and named arguments will be set from the last element. \
            If a named argument is not present in the dictionary , it is \
            assigned the default value, or None if a default value is not \
            provided.

        f: a file-like object or a string containing a file name.  A binary \
            protocol buffer will be written to this file.
        export_params (bool, default True): if True, all parameters will
            be exported.
        verbose (bool, default False): if True, prints a description of the
            model being exported to stdout.
        training (enum, default TrainingMode.EVAL):
            * ``TrainingMode.EVAL``: export the model in inference mode.
            * ``TrainingMode.PRESERVE``: export the model in inference mode \
            if model.training is False and in training mode if model.training \
            is True.
            * ``TrainingMode.TRAINING``: export the model in training mode. \
            Disables optimizations which might interfere with training.
        input_names (list of str, default empty list): names to assign to the
            input nodes of the graph, in order.
        output_names (list of str, default empty list): names to assign to the
            output nodes of the graph, in order.
        operator_export_type (enum, default ONNX_FALLTHROUGH):

            * ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops
              (in the default opset domain).

            * ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops
              to standard ONNX ops in the default opset domain.

            * ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the TorchScript \  # noqa E501
            namespace "aten") are exported as ATen ops.

            * ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each \
            ATen op (in the TorchScript namespace "aten") as a regular ONNX \
            op. If we are unable to do so,fall back to exporting an ATen op.
        opset_version (int, default 11): by default we export the model to the
            opset version of the onnx submodule.
        do_constant_folding (bool, default False): Apply the constant-folding \
            optimization. Constant-folding will replace some of the ops that \
            have all constant inputs with pre-computed constant nodes.
        example_outputs (T or a tuple of T, where T is Tensor or convertible \
            to Tensor, default None):
            Must be provided when exporting a ScriptModule or ScriptFunction,\
             ignored otherwise. Used to determine the type and shape of the \
            outputs without tracing the execution of the model. A single \
            object is treated as equivalent to a tuple of one element.
        dynamic_axes (dict<string, dict<int, string>> or dict<string, list(int)>, \  # noqa E501
             default empty dict):
            By default the exported model will have the shapes of all input \
            and output tensors set to exactly match those given in ``args`` \
            (and ``example_outputs`` when that arg is required). To specify \
            axes of tensors as dynamic (i.e. known only at run-time), set \
            ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be \
            provided in ``input_names`` or ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and \
            values are axis names. If a list, each element is an axis index.

        keep_initializers_as_inputs (bool, default None): If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by
            backends/runtimes.

        custom_opsets (dict<str, int>, default empty dict): A dictionary to indicate  # noqa E501

            A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in\
             this dictionary, the opset version is set to 1.

        enable_onnx_checker (bool, default True): If True the onnx model \
        checker will be run as part of the export, to ensure the exported \
        model is a valid ONNX model.
    """
    if not (operator_export_type == OperatorExportTypes.ONNX_FALLTHROUGH):
        warnings(
            f"Because some Operations are not supported by ONNX, it may "
            f"fail when using `operator_export_type ={operator_export_type}`."
            f"If an error occurs, please try to use `OperatorExportTypes.ONNX_FALLTHROUGH`"  # noqa E501
        )

    with _redirect_stderr():
        torch.onnx.export(
            model,
            args,
            f,
            export_params=export_params,
            keep_initializers_as_inputs=True,
            verbose=verbose,
            training=training,
            input_names=input_names,
            output_names=output_names,
            operator_export_type=operator_export_type,
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
            enable_onnx_checker=enable_onnx_checker,
            dynamic_axes=dynamic_axes,
            example_outputs=example_outputs,
        )


def export_quantized_onnx(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    opset_version=None,
    do_constant_folding=True,
    example_outputs=None,
    dynamic_axes=None,
    keep_initializers_as_inputs=None,
    custom_opsets=None,
):
    r"""
    Export a quantized model into ONNX format.
    Args are same with torch.onnx.export
    """

    assert not (
        isinstance(model, torch.jit.ScriptModule)
        or isinstance(model, torch.jit.ScriptFunction)
    ), (
        "{} is a ScriptModule or ScriptFunction!!".format(model._get_name())
        + " Only support export quantized torch.nn.Module"
    )

    if not (operator_export_type == OperatorExportTypes.ONNX_FALLTHROUGH):
        warnings(
            f"Because some torch Operations are not supported by ONNX, it may "
            f"fail when using `operator_export_type ={operator_export_type}`."
            f"If an error occurs, please try to use `OperatorExportTypes.ONNX_FALLTHROUGH`"  # noqa E501
        )

    with _redirect_stderr():
        torch.onnx.export(
            model,
            args,
            f,
            export_params=export_params,
            verbose=verbose,
            training=training,
            input_names=input_names,
            output_names=output_names,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            example_outputs=example_outputs,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            custom_opsets=custom_opsets,
        )
