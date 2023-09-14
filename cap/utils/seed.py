# Copyright (c) Changan Auto. All rights reserved.

import logging
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from cap.utils.distributed import rank_zero_only
from cap.utils.logger import MSGColor, format_msg

logger = logging.getLogger(__name__)

__all__ = ["seed_everything", "seed_training"]


def seed_everything(seed: int):  # noqa: D205,D400
    """
    Set seed for pseudo-random number generators in:
    pytorch, numpy, python.random.

    Args:
        seed: the integer value seed for global random state.
    """

    # so users can verify the seed is properly set in distributed training.
    logger.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def seed_training(seed: int):  # noqa: D205,D400
    """Set seed for pseudo-random number generators in:
    pytorch, numpy, python.random, set cudnn state as well.

    Args:
        seed: the integer value seed for global random state.
    """
    seed_everything(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    rank_zero_only(
        logger.warning(
            format_msg(
                "You have chosen to seed training. "
                "This will turn on the CUDNN deterministic and turn off CUDNN "
                "benchmark, which can slow down your training considerably! "
                "You may see unexpected behavior when restarting "
                "from checkpoints.",
                MSGColor.RED,
            )
        )
    )
