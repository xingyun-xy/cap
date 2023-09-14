import warnings
from collections import Counter
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from changan_plugin_pytorch.dtype import (
    get_horizon_quant_dtype,
    horizon_quant_dtype,
    qinfo,
)
from torch.distributed import ReduceOp
from torch.jit.annotations import Optional
from torch.quantization.observer import ObserverBase

from .misc import pow_quantization


@torch.jit.script
def compute_scale_symmetric(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: torch.Tensor,
    pow_quantization: bool,
):
    scale = (
        torch.max(-min_val, max_val)
        .clamp_min(0)
        .div(float(quant_max - quant_min) / 2)
        .clamp_min(eps)
    )
    if pow_quantization:
        scale = 1 / 2 ** (torch.floor((-1) * torch.log2(scale)).clamp(1, 14))
    return scale


@torch.jit.script
def compute_moving_average(
    old_min: torch.Tensor,
    old_max: torch.Tensor,
    current_min: torch.Tensor,
    current_max: torch.Tensor,
    averaging_constant: float,
    inplace: bool,
):
    if inplace:
        old_min[:] = old_min + averaging_constant * (current_min - old_min)
        old_max[:] = old_max + averaging_constant * (current_max - old_max)
        return old_min, old_max
    else:
        min_val = old_min + averaging_constant * (current_min - old_min)
        max_val = old_max + averaging_constant * (current_max - old_max)
        return min_val, max_val


class _ObserverBase(ObserverBase):
    _version = 3

    eps: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=None,
        quant_max=None,
        is_sync_quantize=False,
        factory_kwargs=None,
    ) -> None:
        dtype = get_horizon_quant_dtype(dtype)

        assert (
            dtype in horizon_quant_dtype
        ), "Default Observer only works for qint8, qint16 and qint32 data type"
        assert qscheme in (
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
        ), "only support per_tensor_symmetric and per_channel_symmetric qscheme"  # noqa
        assert (
            quant_min is None or type(quant_min) == int
        ), "quant_min should be None or int type"
        assert (
            quant_max is None or type(quant_max) == int
        ), "quant_max should be None or int type"
        assert type(is_sync_quantize) == bool, "is_sync_quantize must be bool"

        super(_ObserverBase, self).__init__(dtype)

        self.is_sync_quantize = is_sync_quantize
        self.qscheme = qscheme

        if (quant_min is not None) and (quant_max is not None):
            assert (
                quant_min < quant_max
            ), "qmin must be strictly less than qmax for user-specified quantization range."  # noqa
            assert (
                quant_min <= 0 <= quant_max
            ), "Used-specified quantization range must include 0."
            assert qinfo(dtype).min <= quant_min, "quant_min out of bound"
            assert quant_max <= qinfo(dtype).max, "quant_max out of bound"
            self.quant_min, self.quant_max = quant_min, quant_max
        else:
            self.quant_min, self.quant_max = (
                qinfo(self.dtype).min,
                qinfo(self.dtype).max,
            )

        self.pow_quantization = pow_quantization()

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer(
            "eps",
            torch.tensor([torch.finfo(torch.float32).eps], **factory_kwargs),
        )
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))

    # @torch.jit.export
    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + "eps"] = eps

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                # if ndim=0, make it ndim=1
                state_dict[key] = state_dict[key].reshape(-1)

                val = state_dict[key]

                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == "min_val" and hasattr(self, "min_val"):
                    self.min_val.resize_(val.shape)
                elif hasattr(self, "max_val"):
                    self.max_val.resize_(val.shape)

        super(_ObserverBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams. Returning default scale and zero point"  # noqa
            )
            return torch.tensor([1.0]), torch.tensor([0])

        if not torch.jit.is_scripting():
            if self.is_sync_quantize and dist.is_initialized():
                dist.all_reduce(min_val, op=ReduceOp.MIN)
                dist.all_reduce(max_val, op=ReduceOp.MAX)

        scale = compute_scale_symmetric(
            min_val,
            max_val,
            self.quant_min,
            self.quant_max,
            self.eps,
            self.pow_quantization,
        )

        # zero_point = torch.zeros_like(scale, dtype=torch.int64)
        zero_point = None

        return scale, zero_point

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)


class MinMaxObserver(_ObserverBase):
    def __init__(
        self,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=None,
        quant_max=None,
        is_sync_quantize=False,
        factory_kwargs=None,
    ) -> None:
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "only support per_tensor_symmetric qscheme"
        super(MinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)

        min_val_cur, max_val_cur = torch._aminmax(x)

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            min_val, max_val = min_val_cur.reshape(-1), max_val_cur.reshape(-1)
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)

        self.min_val, self.max_val = min_val, max_val
        return x_orig


class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: Quantized data type
        qscheme: Quantization scheme to be used, only support
                 per_tensor_symmetric scheme
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value.
        quant_max: Maximum quantization value.
        is_sync_quantize: Whether use sync quantize
        factory_kwargs: Arguments for register data buffer
    """

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=None,
        quant_max=None,
        is_sync_quantize=False,
        factory_kwargs=None,
    ) -> None:

        assert isinstance(averaging_constant, (int, float)), (
            f"averaging_constant should be float or int type,"
            f" but get {type(averaging_constant)}"
        )
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "only support per_tensor_symmetric qscheme"
        super(MovingAverageMinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        self.averaging_constant = float(averaging_constant)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)

        min_val_cur, max_val_cur = torch._aminmax(x)

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val, self.max_val = min_val_cur.reshape(
                -1
            ), max_val_cur.reshape(-1)
        else:
            (self.min_val, self.max_val,) = compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur.reshape(-1),
                max_val_cur.reshape(-1),
                self.averaging_constant,
                True,  # inplace
            )

        return x_orig


class PerChannelMinMaxObserver(_ObserverBase):
    def __init__(
        self,
        ch_axis=0,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=None,
        quant_max=None,
        is_sync_quantize=False,
        factory_kwargs=None,
    ) -> None:
        assert type(ch_axis) == int, "ch_axis should be int type"
        assert (
            qscheme == torch.per_channel_symmetric
        ), "only support per_channel_symmetric qscheme"
        super(PerChannelMinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        self.ch_axis = ch_axis

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)

        if self.ch_axis != 0:
            x = x.transpose(0, self.ch_axis)
        min_val_cur, max_val_cur = torch._aminmax(x.flatten(start_dim=1), 1)

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val, self.max_val = min_val_cur, max_val_cur
        else:
            self.min_val = torch.min(min_val_cur, self.min_val)
            self.max_val = torch.max(max_val_cur, self.max_val)

        return x_orig

    def _load_from_state_dict(
        self,
        state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        # buffers has been renamed from min/max_vals to min/max_val
        buffer_name_mapping = {"min_vals": "min_val", "max_vals": "max_val"}
        for old_name in buffer_name_mapping:
            k = prefix + old_name
            if k in state_dict:
                v = state_dict.pop(k)
                state_dict[prefix + buffer_name_mapping[old_name]] = v

        super(PerChannelMinMaxObserver, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # @torch.jit.export
    def _load_from_state_dict_script(
        self,
        state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):

        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used, Only support
                 per_channel_symmetric
        quant_min: Minimum quantization value.
        quant_max: Maximum quantization value.
        is_sync_quantize: whether use sync quantize
        factory_kwargs: Arguments for register data buffer
    """

    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=None,
        quant_max=None,
        is_sync_quantize=False,
        factory_kwargs=None,
    ) -> None:
        assert isinstance(averaging_constant, (int, float)), (
            f"averaging_constant should be float or int type,"
            f" but get {type(averaging_constant)}"
        )

        super(MovingAveragePerChannelMinMaxObserver, self).__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        self.averaging_constant = float(averaging_constant)

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach().to(self.min_val.dtype)

        if self.ch_axis != 0:
            x = x.transpose(0, self.ch_axis)
        min_val_cur, max_val_cur = torch._aminmax(x.flatten(start_dim=1), 1)

        if self.min_val.numel() == 0 or self.max_val.numel() == 0:
            self.min_val, self.max_val = min_val_cur, max_val_cur
        else:
            (self.min_val, self.max_val,) = compute_moving_average(
                self.min_val,
                self.max_val,
                min_val_cur,
                max_val_cur,
                self.averaging_constant,
                True,  # inplace
            )

        return x_orig


class ClipObserver(_ObserverBase):
    r"""
    This observer uses the tensor min/max statistics to compute the
    quantization parameters. The module records the running minimum and
    maximum of incoming tensors, if the runing minimum is greater the
    designated min value, the statistical minimum result is runing minimum,
    otherwise is the designated min value.And if the running minumum is less
    than the designated xmax, the statistical maxmum is running maxmum,
    otherwise is the designated max value. And uses this statistic to compute
    the quantization parameters.

    Args:
        xmin: Lower bound of statistical minimum
        xmax: Upper bound of statistical maximum
        dtype: Quantized data type
    """

    setted_min: torch.Tensor
    setted_max: torch.Tensor

    def __init__(
        self,
        xmin=-1.0,
        xmax=1.0,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=None,
        quant_max=None,
        is_sync_quantize=False,
        factory_kwargs=None,
    ):
        assert xmin <= xmax, "xmin must less than or equal to xmax"
        assert (
            qscheme == torch.per_tensor_symmetric
        ), "only support per_tensor_symmetric qscheme"
        super(ClipObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)

        self.min_val = torch.tensor([float(xmin)], **factory_kwargs)
        self.max_val = torch.tensor([float(xmax)], **factory_kwargs)
        self.register_buffer(
            "setted_min", torch.tensor([float(xmin)], **factory_kwargs)
        )
        self.register_buffer(
            "setted_max", torch.tensor([float(xmax)], **factory_kwargs)
        )

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``
            and clip the minimum and maximum into [xmin, xmax]
        ."""
        x = x_orig.detach().to(self.min_val.dtype)

        min_val_cur, max_val_cur = torch._aminmax(x)
        self.min_val = torch.max(min_val_cur, self.setted_min)
        self.max_val = torch.min(max_val_cur, self.setted_max)

        return x_orig


class FixedScaleObserver(_ObserverBase):
    r"""
    This observer always return a fixed scale and zero_point regardless of
    input data.

    Args:
        scale (float): Fixed scale value.
        zero_point (int): Fixed zero_point value.
        dtype: Quantized data type.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        scale,
        zero_point=0,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=None,
        quant_max=None,
        is_sync_quantize=False,
        factory_kwargs=None,
    ):
        assert scale > 0, "scale must bigger than 0"
        super(FixedScaleObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            quant_min=quant_min,
            quant_max=quant_max,
            is_sync_quantize=is_sync_quantize,
            factory_kwargs=factory_kwargs,
        )

        del self.min_val
        del self.max_val
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.float).reshape(-1)
        )
        self.register_buffer(
            "zero_point",
            torch.tensor(zero_point, dtype=torch.long).reshape(-1),
        )

    def forward(self, x_orig):
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        return "scale={}, zero_point={}".format(self.scale, self.zero_point)


class CalibObserver(torch.nn.Module):
    """Unified histogram calibrator

    Histogram will be only collected once. compute_amax() performs entropy,
    percentile, or mse calibration based on arguments

    Args:
        num_bits: An integer. Number of bits of quantization.
        axis: A tuple. see QuantDescriptor.
        unsigned: A boolean. using unsigned quantization.
        num_bins: An integer. Number of histograms bins. Default 2048.
        skip_zeros: A boolean. If True, skips zeros when collecting data for
        method: A string. Compute amax method, percentile or entropy
        percentile: A float. index percentile of histrogram
    """

    def __init__(
        self,
        num_bits,
        axis,
        unsigned,
        num_bins=2048,
        skip_zeros=False,
        method="percentile",
        percentile=99.99,
    ):
        super(CalibObserver, self).__init__()
        self._num_bits = num_bits
        self._num_bins = num_bins
        self._skip_zeros = skip_zeros
        self._unsigned = unsigned
        self._calib_bin_edges = None
        self._calib_hist = None
        self._method = method
        self._device = None
        self._percentile = percentile
        assert self._method in ["entropy", "percentile"]
        if axis is not None:
            raise NotImplementedError(
                "Calibrator histogram collection only supports per tensorscaling"  # noqa
            )

    def collect(self, x):
        """Collect histogram"""
        self._device = x.device
        with torch.no_grad():
            if torch.min(x) < 0.0:
                x = x.abs()
            x = x.float()

            if self._skip_zeros:
                x = x[torch.where(x != 0)]

            x_max = x.max()
            if self._calib_bin_edges is None and self._calib_hist is None:
                # TODO: replace with `torch.histogram` in torch 1.10
                self._calib_hist = torch.histc(
                    x, bins=self._num_bins, min=0, max=x_max
                )
                self._calib_bin_edges = torch.linspace(
                    0, x_max, self._num_bins + 1, device=self._device
                )
            else:
                if x_max > self._calib_bin_edges[-1]:
                    width = self._calib_bin_edges[1] - self._calib_bin_edges[0]
                    self._num_bins = int((x_max / width).ceil().item())
                    self._calib_bin_edges = torch.arange(
                        0, x_max + width, width, device=self._device
                    )
                    # TODO: replace with `torch.histogram` in torch 1.10
                    hist = torch.histc(
                        x,
                        bins=self._num_bins,
                        min=0,
                        max=self._calib_bin_edges[-1],
                    )
                    hist[: self._calib_hist.numel()] += self._calib_hist
                    self._calib_hist = hist

    def forward(self, x):
        self.collect(x.detach())
        return x

    def reset(self):
        """Reset the collected histogram"""
        self._calib_bin_edges = None
        self._calib_hist = None

    def compute_amax(
        self,
        *,
        stride: int = 1,
        start_bin: int = 128,
    ):
        """Compute the amax from the collected histogram

        Args:
        Keyword Arguments:
            stride: An integer. Default 1
            start_bin: An integer. Default 128
            percentils: A float number between [0, 100]. Default 99.99.

        Returns:
            amax: a tensor
        """
        if self._method == "entropy":
            calib_amax = self._compute_amax_entropy(
                self._calib_hist,
                self._calib_bin_edges,
                self._num_bits,
                self._unsigned,
                stride,
                start_bin,
            )
        elif self._method == "percentile":
            calib_amax = self._compute_amax_percentile(
                self._calib_hist, self._calib_bin_edges, self._percentile
            )
        else:
            raise TypeError(
                "Unknown calibration method {}".format(self._method)
            )
        if calib_amax is None:
            warnings.warn("Compute amax failed")
            return None
        return calib_amax.reshape(-1)

    def __str__(self):
        s = "CalibObserver("
        if self._calib_bin_edges is None:
            bin_edge_str = "None"
        else:
            bin_edge_str = "[{:.3f}, ..., {:.3f}]({})".format(
                self._calib_bin_edges[0],
                self._calib_bin_edges[-1],
                len(self._calib_bin_edges),
            )
        s += "calib_bin_edges={})".format(bin_edge_str)
        return s

    def __repr__(self):
        s = "CalibObserver("
        s += super(CalibObserver, self).__repr__()
        s += " calib_bin_edges={_calib_bin_edges}"
        s += " calib_hist={_calib_hist})"
        return s.format(**self.__dict__)

    def _entropy_torch(self, pk, qk=None, axis=0):
        """This is the pytorch implementation of `scipy.stats.entropy`,
         and it is as close as possible to the original implementation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

        Args:
            pk (torch.Tensor): Defines the (discrete) distribution. pk[i] is
             the (possibly unnormalized) probability of event i.
            qk (torch.Tensor): Sequence against which the relative
             entropy is computed. Should be in the same format as pk.
            axis (int): The axis along which the entropy is calculated.
                Default is 0.

        Returns:
            torch.Tensor: The calculated entropy.
        """

        pk = 1.0 * pk / torch.sum(pk, dim=axis, keepdim=True)

        if qk is None:
            # Note: `torch.special,entr()` requires torch.__version__>=1.9.0
            vec = torch.special.entr(pk)
        else:
            if qk.shape != pk.shape:
                raise ValueError("qk and pk must have same shape.")
            qk = 1.0 * qk / torch.sum(qk, dim=axis, keepdim=True)
            vec = -(qk) * (torch.special.entr(pk / qk))
        ret = torch.sum(vec, dim=axis)

        return ret

    def _compute_amax_entropy(
        self,
        calib_hist,
        calib_bin_edges,
        num_bits,
        unsigned,
        stride=1,
        start_bin=128,
    ):
        """Returns amax that minimizes KL-Divergence of the collected
        histogram
        """

        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        if self._device is None:
            self._device = calib_bin_edges.device

        def _normalize_distr(distr):
            summ = torch.sum(distr)
            if summ != 0:
                distr = distr / summ

        bins = calib_hist[:]
        bins[0] = bins[1]

        total_data = torch.sum(bins)

        divergences = []
        arguments = []

        # we are quantizing to 128 values + sign if num_bits=8
        nbins = 1 << (num_bits - 1 + int(unsigned))

        starting = start_bin
        stop = len(bins)

        new_density_counts = torch.zeros(
            nbins, dtype=torch.float64, device=self._device
        )

        for i in range(starting, stop + 1, stride):
            new_density_counts.fill_(torch.tensor(0))
            space = torch.linspace(0, i, nbins + 1, device=self._device)

            # TODO: after pytorch supporte `torch.digitize`, replace it.
            digitized_space = (
                torch.bucketize(
                    torch.arange(i, device=self._device), space, right=True
                )
                - 1
            )

            digitized_space[bins[:i] == 0] = -1

            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density_counts[digitized] += bins[idx]

            counter = Counter(digitized_space.cpu().numpy())
            for key, val in counter.items():
                if key != -1:
                    new_density_counts[key] = new_density_counts[key] / val

            new_density = torch.zeros(
                i, dtype=torch.float64, device=self._device
            )
            for idx, digitized in enumerate(digitized_space):
                if digitized.item() != -1:
                    new_density[idx] = new_density_counts[digitized]

            total_counts_new = torch.sum(new_density) + torch.sum(bins[i:])
            _normalize_distr(new_density)

            reference_density = bins[: len(digitized_space)].clone().detach()
            reference_density[-1] += torch.sum(bins[i:])

            total_counts_old = torch.sum(reference_density)
            if (
                round(total_counts_new.item()) != total_data.item()
                or round(total_counts_old.item()) != total_data.item()
            ):
                raise RuntimeError(
                    "Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(  # noqa
                        total_counts_new, total_counts_old, total_data
                    )
                )

            _normalize_distr(reference_density)

            ent = self._entropy_torch(pk=reference_density, qk=new_density)
            divergences.append(ent)
            arguments.append(i)

        divergences = torch.as_tensor(divergences)
        last_argmin = len(divergences) - 1 - torch.argmin(divergences, dim=0)
        calib_amax = calib_bin_edges[last_argmin * stride + starting]

        return calib_amax

    def _compute_amax_percentile(
        self, calib_hist, calib_bin_edges, percentile
    ):
        """Returns amax that clips the percentile fraction of collected data"""

        if percentile < 0 or percentile > 100:
            raise ValueError(
                "Invalid percentile. Must be in range 0 <= percentile <= 100."
            )

        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        total = calib_hist.sum()
        cdf = torch.cumsum(calib_hist / total, dim=0)
        idx = torch.searchsorted(cdf, percentile / 100)
        calib_amax = calib_bin_edges[idx]

        return calib_amax
