import torch
from changan_plugin_pytorch.dtype import qint8
from changan_plugin_pytorch.march import March, get_march
from changan_plugin_pytorch.nn import qat
from changan_plugin_pytorch.qtensor import QTensor

from .functional import multi_scale_roi_align


class RoIAlign(torch.nn.Module):
    """
    Region of Interest (RoI) Align operator described in Mask R-CNN.
    We do center alignment as opencv, this behaviour is
    different from torchvision.ops.RoIAlign.

    Parameters
    ----------
    Same as float version.
    """

    _QAT_MODULE = qat.RoIAlign
    # TODO Add arbitrary sampling_ratio support after avg pool op complete.

    def __init__(
        self,
        output_size,
        spatial_scale=1.0,
        sampling_ratio=1,
        aligned=False,
        interpolate_mode="bilinear",
        out_dtype=qint8,
    ):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.out_dtype = out_dtype
        self.interpolate_mode = interpolate_mode

        if (
            get_march() in (March.BERNOULLI2, March.BERNOULLI)
        ) and self.aligned is not None:
            assert False, (
                "Not support 'aligned' parameter on Bernoulli2 or Bernoulli! "
                + "Bernoulli and Bernoulli2 set roi_w = roi * spatical_scale "
                + "+ 1 and use origin interpolate mode."
            )

    def forward(self, featuremaps, rois):
        """
        Bpu inference forward pass of ~RoIAlign.

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
        if not isinstance(rois, list):
            raise ValueError("RoiAlign only accept roi as List[Tensor]")

        if isinstance(rois[0], QTensor):
            for roi in rois:
                assert roi.q_scale().item() == 0.25, (
                    "invalid input roi scale, "
                    + "we expect 0.25, but receive {}".format(
                        roi.q_scale().item()
                    )
                )
            rois = [roi.int_repr() for roi in rois]

        out = multi_scale_roi_align(
            [featuremaps.int_repr()],
            rois,
            self.output_size,
            [self.spatial_scale],
            self.sampling_ratio,
            self.aligned if self.aligned is not None else False,
            self.interpolate_mode,
            1,
            0,
        )

        return QTensor(out, featuremaps.q_scale(), self.out_dtype)

    @classmethod
    def from_float(cls, mod):
        r"""Create a quantized module from a qat module"""
        assert type(mod) == cls._QAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )

        quantized_mod = cls(
            output_size=mod.output_size,
            spatial_scale=mod.spatial_scale,
            sampling_ratio=mod.sampling_ratio,
            aligned=mod.aligned,
            interpolate_mode=mod.interpolate_mode,
            out_dtype=mod.activation_post_process.dtype,
        )

        return quantized_mod
