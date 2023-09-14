import torch
from changan_plugin_pytorch.qtensor import QTensor
from torch.nn.parameter import Parameter
from torch.quantization.fake_quantize import _is_per_channel
from torch.quantization.observer import _with_args
import warnings
from .fake_quantize import _is_affine
from .observer import (
    FixedScaleObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)


class _LearnableFakeQuantize(torch.quantization.FakeQuantizeBase):
    r"""This is an extension of the FakeQuantize module in fake_quantize.py,
    which supports more generalized lower-bit quantization and support
    learning of the scale and zero point parameters through backpropagation.
    For literature references,
    please see the class _LearnableFakeQuantizePerTensorOp.
    In addition to the attributes in the original FakeQuantize module,
    the _LearnableFakeQuantize module also includes the following attributes
    to support quantization parameter learning.
    * :attr:`channel_len` defines the length of the channel
       when initializing scale and zero point for the per channel case.
    * :attr:`use_grad_scaling` defines the flag for whether the gradients
       for scale and zero point are normalized by the constant,
       which is proportional to the square root of the number of
       elements in the tensor. The related literature justifying
       the use of this particular constant can be found here:
       https://openreview.net/pdf?id=rkgO66VKDS.
    * :attr:`fake_quant_enabled` defines the flag
       for enabling fake quantization on the output.
    * :attr:`static_enabled` defines the flag
       for using observer's static estimation for scale and zero point.
    * :attr:`learning_enabled` defines the flag for enabling backpropagation
       for scale and zero point.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    learning_enabled: torch.Tensor
    eps: torch.Tensor

    def __init__(
        self,
        observer,
        quant_min=None,
        quant_max=None,
        scale=None,
        zero_point=None,
        channel_len=1,
        use_grad_scaling=False,
        **observer_kwargs
    ):
        assert type(channel_len) == int, "channel_len should be int type"
        assert (
            channel_len >= 1
        ), "channel_len should greater than or equal to 1"

        super(_LearnableFakeQuantize, self).__init__()
        # use flags rather than buffer to avoid cuda to cpu copy and
        # speed up forward
        self._fake_quant_enabled = True
        self._observer_enabled = True

        self.activation_post_process = observer(
            quant_min=quant_min,
            quant_max=quant_max,
            **observer_kwargs,
        )
        # get quant minmax from observer where they are properly configured
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = (
            self.activation_post_process.ch_axis
            if hasattr(self.activation_post_process, "ch_axis")
            else -1
        )

        self.is_per_channel = _is_per_channel(self.qscheme)
        self.is_affine = _is_affine(self.qscheme)

        if self.is_per_channel:
            scale_len = channel_len
        else:
            scale_len = 1

        self.use_grad_scaling = use_grad_scaling

        self.scale = Parameter(torch.tensor([1.0] * scale_len))
        self.zero_point = Parameter(
            torch.tensor([0.0] * scale_len),
            requires_grad=self.is_affine,
        )

        self.register_buffer(
            "learning_enabled", torch.tensor([1], dtype=torch.uint8)
        )
        self.register_buffer(
            "eps", torch.tensor([torch.finfo(torch.float32).eps])
        )

        if scale is not None:
            self.set_qparams(scale, zero_point)
        if observer == FixedScaleObserver:
            assert (
                scale is None and zero_point is None
            ), "cannot use FixedScaleObserver while quanti params are provided"
            (
                fixed_scale,
                fixed_zero_point,
            ) = self.activation_post_process.calculate_qparams()
            self.set_qparams(fixed_scale, fixed_zero_point)
            self.disable_param_learning()
            self.param_fixed = True
        else:
            self.param_fixed = False

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0
        self._fake_quant_enabled = enabled

    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None:
        # TODO: this method should called "enable_param_update"
        # now we call it "enable_observer" to be compatible with FakeQuantize
        self.enable_param_learning(enabled)

    @torch.jit.export
    def enable_param_learning(self, enabled: bool = True):
        if self.param_fixed:
            warnings.warn(
                "trying to enable param learning on fixed param, ignored",
                stacklevel=5,
            )
        else:
            self.learning_enabled[0] = 1 if enabled else 0
            self.scale.requires_grad_(enabled)
            self.zero_point.requires_grad_(self.is_affine and enabled)

    @torch.jit.export
    def disable_param_learning(self):
        self.enable_param_learning(False)

    @torch.jit.export
    def calculate_qparams(self):
        scale = self.scale.detach().abs().clamp(min=self.eps)
        zero_point = (
            self.zero_point.detach()
            .round()
            .clamp(self.quant_min, self.quant_max)
            .long()
        )
        return scale, zero_point

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
        v = state_dict.get(prefix + "observer_enabled", None)
        if v is not None:
            self._observer_enabled = v[0].item() == 1

        v = state_dict.get(prefix + "fake_quant_enabled", None)
        if v is not None:
            self._fake_quant_enabled = v[0].item() == 1

        super(_LearnableFakeQuantize, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _check_zero_point(self):
        if not self.is_affine:
            assert torch.all(
                self.zero_point == 0
            ), "zero point must be 0 in symmetric quantization"

    def set_qparams(self, scale, zero_point=None):
        with torch.no_grad():
            self.scale[:] = scale
            if zero_point is not None:
                self.zero_point[:] = zero_point

            # disable the observer since we have init qparams
            self._observer_enabled = False
            self.observer_enabled[:] = 0

            self._check_zero_point()

    def forward(self, X):
        with torch.no_grad():
            if self._observer_enabled:  # type: ignore[index]
                # only use observer for the first forward
                self.activation_post_process(X.detach())
                (
                    _scale,
                    _zero_point,
                ) = self.activation_post_process.calculate_qparams()
                self.scale.copy_(_scale)
                if _zero_point is not None:
                    self.zero_point.copy_(_zero_point)

                self._observer_enabled = False
                self.observer_enabled[:] = 0
            else:
                self.scale.abs_().clamp_(min=self.eps)
                self.zero_point.clamp_(self.quant_min, self.quant_max)

        if self._fake_quant_enabled == 1:
            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0

            if self.is_per_channel:
                # TODO: numerical alignment with quantized
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.ch_axis,
                    self.quant_min,
                    self.quant_max,
                    grad_factor,
                )
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X,
                    self.scale,
                    self.zero_point,
                    self.quant_min,
                    self.quant_max,
                    grad_factor,
                )
        return QTensor(
            data=X,
            scale=self.scale,
            dtype=self.dtype,
            per_channel_axis=self.ch_axis,
        )

    with_args = classmethod(_with_args)


default_8bit_lsq_quant = _LearnableFakeQuantize.with_args(
    observer=MinMaxObserver,
    use_grad_scaling=True,
    dtype="qint8",
)
default_weight_8bit_lsq_quant = _LearnableFakeQuantize.with_args(
    observer=PerChannelMinMaxObserver,
    use_grad_scaling=True,
    dtype="qint8",
    ch_axis=0,
)
default_4bit_lsq_quant = _LearnableFakeQuantize.with_args(
    observer=MinMaxObserver,
    use_grad_scaling=True,
    dtype="qint4",
)
default_uint4_lsq_quant = _LearnableFakeQuantize.with_args(
    observer=MinMaxObserver,
    dtype="quint4",
)
default_weight_4bit_lsq_quant = _LearnableFakeQuantize.with_args(
    observer=PerChannelMinMaxObserver,
    use_grad_scaling=True,
    dtype="qint4",
    ch_axis=0,
)
default_16bit_lsq_quant = _LearnableFakeQuantize.with_args(
    observer=MinMaxObserver,
    use_grad_scaling=True,
    dtype="qint16",
)
default_weight_16bit_lsq_quant = _LearnableFakeQuantize.with_args(
    observer=PerChannelMinMaxObserver,
    use_grad_scaling=True,
    dtype="qint16",
    ch_axis=0,
)
