from functools import wraps

import torch
from changan_plugin_pytorch.qtensor import QTensor
from torch import Tensor

__all__ = ["call_with_hooks", "assert_qtensor_input", "swap_ff_with_horizonff"]


def call_with_hooks(func):
    r"""call Module method with hooks"""
    import torch.utils.hooks as hooks

    @wraps(func)  # retain owner information
    def _call_impl(mod, *input, **kwargs):
        mod._last_called_method_name = func.__name__

        # copy from module._call_impl
        # Do not call functions when jit is used
        full_backward_hooks = []
        if len(mod._backward_hooks) > 0:
            full_backward_hooks, _ = mod._get_backward_hooks()

        # forward pre
        for hook in mod._forward_pre_hooks.values():
            result = hook(mod, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result

        bw_hook = None
        if len(full_backward_hooks) > 0:
            bw_hook = hooks.BackwardHook(mod, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)

        # call func
        result = func(mod, *input, **kwargs)

        for hook in mod._forward_hooks.values():
            hook_result = hook(mod, input, result)
            if hook_result is not None:
                result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)
        return result

    return _call_impl


def assert_qtensor_input(input):
    """Check if all Tensor-like object is QTensor in module input"""
    if isinstance(input, (list, tuple)):
        for x in input:
            assert_qtensor_input(x)
    else:
        if isinstance(input, Tensor) and not isinstance(input, QTensor):
            raise ValueError(
                "module input expect to be QTensor, but receive Tensor"
            )


def swap_ff_with_horizonff(model: torch.nn.Module) -> None:
    r"""
    Swap torch.nn.quantized.FloatFunctional with
    changan_plugin_pytorch.nn.quantized.FloatFunctional,
    which is wrapped with changan fx wrapper
    """
    import changan_plugin_pytorch

    modules_to_swap = []
    for name, module in model.named_children():
        if isinstance(
            module,
            torch.nn.quantized.FloatFunctional,
        ):
            modules_to_swap.append(name)
        else:
            swap_ff_with_horizonff(module)

    for name in modules_to_swap:
        setattr(
            model, name, changan_plugin_pytorch.nn.quantized.FloatFunctional()
        )
