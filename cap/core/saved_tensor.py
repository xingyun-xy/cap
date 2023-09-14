from decimal import localcontext
from distutils.version import LooseVersion

import torch

NEED_TORCH_VERSION = "1.10.2"


def check_support():
    match = LooseVersion(torch.__version__) >= LooseVersion(NEED_TORCH_VERSION)
    return match


class TensorBuffer:
    """
    Save tensor for recompute later.

    Use grad_fn as key, because grad_fn is unique for each grad node
    """

    _saved_tensor_list = {}  # {grad_fn : [t, fn, refcnt]}

    @classmethod
    def add_tensor(cls, t, origin_t, rematerial_fn):
        grad_fn = t.grad_fn
        dtype = t.dtype
        if grad_fn not in cls._saved_tensor_list:
            cls._saved_tensor_list[grad_fn] = [origin_t, rematerial_fn, dtype]
        else:
            # use assert to raise invalid saving.
            raise RuntimeError("has already saved for tensor")

    @classmethod
    def get_tensor(cls, k):
        if k not in cls._saved_tensor_list:
            return k

        data = cls._saved_tensor_list[k][0]  # get origin input
        fn = cls._saved_tensor_list[k][1]  # get recompute fn
        dtype = cls._saved_tensor_list[k][2]  # get dtype

        if dtype != torch.float32:
            autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype)
        else:
            autocast = localcontext()

        # recompute output tensor without grad
        with torch.no_grad():
            with autocast:
                res = fn(data)
        cls._saved_tensor_list.pop(k)
        return res


def get_fn_conv_bn(conv, bn, act=None):
    """
    Return a function object that use savedtensor feature for conv+bn+act.

    Set conv as func1 and bn+act as func2.
    The func1 runs as usual, func2 run with context of saved tensor hooks.
    """
    if not check_support():
        raise RuntimeError("not support using savedtensor for conv-bn")

    def _fn1(x):
        return conv(x)

    def _fn2(x):
        x = bn(x)
        if act is not None:
            x = act(x)
        return x

    conv_grad_fn = "CudnnConvolutionBackward0"
    check_grad_next = False
    if conv.bias is not None:
        check_grad_next = True

    def do_conv_bn(data):
        def pack(t):
            if t.grad_fn is not None:
                if (
                    check_grad_next
                    and isinstance(t.grad_fn.next_functions[0], (list, tuple))
                    and t.grad_fn.next_functions[0][0] is not None
                    and t.grad_fn.next_functions[0][0].name() == conv_grad_fn
                ):
                    TensorBuffer.add_tensor(t, data, _fn1)
                    return t.grad_fn
                elif (
                    t.grad_fn is not None
                    and not check_grad_next
                    and t.grad_fn.name() == conv_grad_fn
                ):
                    TensorBuffer.add_tensor(t, data, _fn1)
                    return t.grad_fn
            return t

        def unpack(t):
            t = TensorBuffer.get_tensor(t)
            return t

        out = _fn1(data)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = _fn2(out)
        return out

    return do_conv_bn
