from .decoder import FCNDecoder
from .head import DepthwiseSeparableFCNHead, FCNHead
from .target import FCNTarget

__all__ = ["FCNHead", "DepthwiseSeparableFCNHead", "FCNTarget", "FCNDecoder"]
