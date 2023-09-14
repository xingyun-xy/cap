# Copyright (c) Changan Auto. All rights reserved.
import logging

__all__ = ["pil_loader"]

logger = logging.getLogger(__name__)


def pil_loader(data_path, mode="RGB", size=None, timeout=60):
    """
    Load image using PIL, open path as file to avoid ResourceWarning.

    # (https://github.com/python-pillow/Pillow/issues/835)

    """
    import timeout_decorator
    from PIL import Image

    @timeout_decorator.timeout(timeout)
    def _pil_loader(data_path, io_mode):
        fid = open(data_path, io_mode)
        with Image.open(fid) as img:
            if size is None:
                img = img.convert(mode)
            else:
                img.draft(mode, size)
                img.load()
        fid.close()
        return img

    try:
        img = _pil_loader(data_path, "rb")
        return img
    except (timeout_decorator.TimeoutError, FileNotFoundError) as e:
        if isinstance(e, timeout_decorator.TimeoutError):
            logger.info(f"read {data_path} timeout > {timeout}sec")
        elif isinstance(e, FileNotFoundError):
            logger.info(f"{data_path} FileNotFoundError")
        raise FileNotFoundError
