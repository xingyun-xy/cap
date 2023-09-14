import warnings

import torch
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input
from torch.nn.modules.utils import _pair
from torchvision import ops as nnv

from .functional import roi_align_list, roi_align_tensor


class RoIAlign(torch.nn.Module):
    """
    Region of Interest (RoI) Align operator described in Mask R-CNN.
    We do center alignment as opencv, this behaviour is
    different from torchvision.ops.RoIAlign.

    Parameters
    ----------
    Same as float version. Except that we add 'interpolate_mode' for mode
    selection when using MultiScaleRoIAlign.
    Note that we do not support float 'nearest' RoIAlign convert to qat
    """

    _FLOAT_MODULE = nnv.RoIAlign

    # TODO Add arbitrary sampling_ratio support after avg pool op complete.

    def __init__(
        self,
        output_size,
        spatial_scale=1.0,
        sampling_ratio=1,
        aligned=False,
        interpolate_mode="bilinear",
        qconfig=None,
    ):
        super(RoIAlign, self).__init__()

        assert qconfig, "qconfig must be provided for QAT module"
        assert (
            qconfig.activation
        ), "qconfig must have member activation for qat.RoIAlign"
        assert interpolate_mode == "bilinear", "only support 'bilinear' mode"
        self.qconfig = qconfig
        self.activation_post_process = self.qconfig.activation()
        self.activation_post_process.disable_observer()

        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.interpolate_mode = interpolate_mode
        # This option is only used in qat.MultiScaleRoIAlign
        self._allow_tensor_roi = False

        self._check_init()

    def _check_init(self):
        out_height, out_width = _pair(self.output_size)
        assert out_height > 0 and out_width > 0, "output size must be positive"
        assert self.sampling_ratio == 1, "only support sampling_ratio = 1"

        if (
            get_march() in (March.BERNOULLI2, March.BERNOULLI)
        ) and self.aligned is not None:

            assert False, (
                "Not support 'aligned' parameter on Bernoulli2 and Bernoulli! "
                + "Bernoulli and Bernoulli2 set roi_w = roi * spatical_scale "
                + "+ 1 and use origin interpolate mode."
            )

    def forward(self, featuremaps, rois):
        """
        Quanti forward pass of ~RoIAlign.

        Args:
            featuremaps (QTensor): Featuremap.
            rois (List[Tensor[L, 4]]):
                The box coordinates in (x1, y1, x2, y2) format where the
                regions will be taken from. Each Tensor will correspond to
                the boxes for an element i in a batch.

                When march = bernoulli2, rois should only be produced by
                DetectionPostProcessV1

        Returns:
            QTensor: Pooled featuremap associate with rois.
        """
        assert_qtensor_input(featuremaps)

        roi_quantized = False

        self.activation_post_process.set_qparams(featuremaps.q_scale())

        if isinstance(rois, list):
            if isinstance(rois[0], QTensor):
                roi_quantized = True
                for roi in rois:
                    assert roi.q_scale().item() == 0.25, (
                        "invalid input roi scale, "
                        + "we expect 0.25, but receive {}".format(
                            roi.q_scale().item()
                        )
                    )
                rois = [roi.as_subclass(torch.Tensor) for roi in rois]
                # check rois have no grad
                # changan_plugin_pytorch/issues/67
                for i, roi in enumerate(rois):
                    if roi.requires_grad:
                        warnings.warn(
                            "rois should not have grad. "
                            "This grad will be ignored..."
                        )
                        rois[i] = roi.detach()

            out = roi_align_list(
                featuremaps.as_subclass(torch.Tensor),
                rois,
                _pair(self.output_size),
                self.spatial_scale,
                self.sampling_ratio,
                self.aligned if self.aligned is not None else False,
                self.interpolate_mode,
                featuremaps.q_scale(),
                featuremaps.q_zero_point(),
                featuremaps.dtype,
                roi_quantized,
            )
        else:
            if not self._allow_tensor_roi:
                raise ValueError("RoiAlign only accept roi as List[Tensor]")
            if isinstance(rois, QTensor):
                roi_quantized = True
                assert rois.q_scale().item() == 0.25, (
                    "invalid input roi scale, "
                    + "we expect 0.25, but receive {}".format(
                        rois.q_scale().item()
                    )
                )
                rois = rois.as_subclass(torch.Tensor)

            # check rois have no grad
            # changan_plugin_pytorch/issues/67
            if rois.requires_grad:
                warnings.warn(
                    "rois should not have grad. "
                    "This grad will be ignored..."
                )
                rois = rois.detach()

            out = roi_align_tensor(
                featuremaps.as_subclass(torch.Tensor),
                rois,
                _pair(self.output_size),
                self.spatial_scale,
                self.sampling_ratio,
                self.aligned if self.aligned is not None else False,
                self.interpolate_mode,
                featuremaps.q_scale(),
                featuremaps.q_zero_point(),
                featuremaps.dtype,
                roi_quantized,
            )

        return self.activation_post_process(out)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qat_mod = cls(
            output_size=mod.output_size,
            spatial_scale=mod.spatial_scale,
            sampling_ratio=mod.sampling_ratio,
            aligned=mod.aligned,
            interpolate_mode=(
                mod.interpolate_mode
                if hasattr(mod, "interpolate_mode")
                else "bilinear"
            ),
            qconfig=mod.qconfig,
        )

        return qat_mod
