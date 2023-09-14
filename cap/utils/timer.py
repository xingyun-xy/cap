import functools
import logging
import time

logger = logging.getLogger(__name__)

__all__ = ["Timer", "BytesTimer"]


class Timer:
    """
    Class for timing execution speed of function.

    Args:
        name: name of timer
        per_iters: cycles of logging speed info.
    """

    def __init__(self, name, per_iters=1000):
        self.name = name
        self.per_iters = per_iters
        self.reset()

    def reset(self):
        self.iter_num = 0
        self.elapsed_time = 0
        self.tic_time = None

    def tic(self):
        self.tic_time = time.time()

    def toc(self):
        self.elapsed_time += time.time() - self.tic_time
        self.iter_num += 1
        if self.iter_num > self.per_iters:
            self._log()
            self.reset()

    def _log(self):
        logger.info(
            "speed of {} is {} iters/sec".format(
                self.name, self.iter_num / self.elapsed_time
            )
        )

    def timeit(self, func):
        @functools.wraps(func)
        def with_timer(*args, **kwargs):
            self.tic()
            ret = func(*args, **kwargs)
            self.toc()
            return ret

        return with_timer


class BytesTimer(Timer):
    """
    Class for timing execution speed of read raw data.

    Args:
        name: name of timer
        per_iters: cycles of logging speed info.
    """

    def __init__(self, name="reading raw data", per_iters=1000):
        super().__init__(name, per_iters)

    def reset(self):
        super().reset()
        self.read_mbytes = 0

    def toc(self, mbytes):
        self.read_mbytes = mbytes
        super().toc()

    def timeit(self, func):
        @functools.wraps(func)
        def with_timer(*args, **kwargs):
            self.tic()
            ret = func(*args, **kwargs)
            assert isinstance(ret, bytes)
            mbytes = len(ret) / 1024.0 / 1024.0
            self.toc(mbytes)
            return ret

        return with_timer

    def _log(self):
        logger.info(
            "speed of {} is {} mbytes/sec, {} record/sec".format(
                self.name,
                self.read_mbytes / self.elapsed_time,
                self.iter_num / self.elapsed_time,
            )
        )
