# Copyright (c) Changan Auto. All rights reserved.
import logging
import multiprocessing
import os
import time

from torch.multiprocessing import ProcessContext

from .pack_type import PackTypeMapper

__all__ = ["Packer"]

logger = logging.getLogger(__name__)


def _read_worker(packer, data_idx, queue):
    """
    Read data and put into queue by idx of q_in.

    Processed data by packer.pack_data.

    Args:
        packer (Packer): Instantiated packer.
        data_idx (list): Packing index.
        queue (multiprocess.Queue): Queue for packing data.

    """

    try:
        for idx in data_idx:
            block = packer.pack_data(idx)
            if block is not None:
                queue.put((idx, block))
    except Exception as e:
        queue.put(None)
        raise e


def _write_worker(packer, queue):
    """
    Write processed data into packer.dataset.

    Args:
        packer (Packer): Instantiated packer.
        queue (multiprocess.Queue): Queue for packing data.
    """

    idx = 0
    pre_time = time.time()
    while True:
        data = queue.get()
        if data is None:
            # try to write length for dataset
            packer._write_length(idx)
            break

        q_idx, q_data = data

        packer._write(idx, q_data)
        idx += 1
        if idx % 100 == 0:
            cur_time = time.time()
            logger.info(f"time: {cur_time - pre_time}, count: {idx}")
            pre_time = cur_time


class Packer(object):
    """
    Abstact class of Packing data into target packtype.

    Packer is the recommended base class for being inherited.
    Focus on packer env, read and write.

    Subclass need to override :py:func:`pack_data`.

    Args:
        uri (str): Path to the packing file.
        max_data_num (int): Num of data for packing.
        pack_type (str): The target pack type.
        num_workers (int): Num workers for reading original data.
            while num_workers <= 0 means pack by single process.
            num_workers >= 1 mean pack by num_workers process.
        queue_maxsize (int): Max size of queue.
        start_method (str): Start method for multiprocess.
        **kwargs (dict): kwargs for pack type.
    """

    def __init__(
        self,
        uri: str,
        max_data_num: int,
        pack_type: str,
        num_workers: int,
        queue_maxsize: int = 1024,
        start_method: str = "fork",
        **kwargs,
    ):
        self.num_blocks = max_data_num
        self.num_workers = num_workers
        self.queue_maxsize = queue_maxsize
        self.start_method = start_method

        if not os.path.exists(uri):
            os.makedirs(uri)

        self.target_pack_type = pack_type.lower()
        assert self.target_pack_type in [
            "lmdb",
            "mxrecord",
        ], f"lmdb, mxrecord are supported packtypes, except {pack_type}"
        PackType = PackTypeMapper[self.target_pack_type]

        self.pack_file = PackType(uri=uri, writable=True, **kwargs)

    def __call__(self):
        if self.num_workers <= 0:
            logger.info("Use single process for packing.")
            self._sp_process()
        else:
            logger.info(f"Use {self.num_workers} processes for packing.")
            self._mp_process()

    def pack_data(self, idx):
        """
        Read orginal data from Folder with some process.

        Args:
            idx (int): Idx for reading.

        Returns:
            Processed data for pack.
        """

        raise NotImplementedError

    def _write(self, idx, data):
        """
        Write data to target format file.

        Args:
            idx (int): Idx for writing.
            data : Processed data for writing to target PackType.
        """
        self.pack_file.write(idx, data)

    def _write_length(self, num):
        """
        Write max length to target format file.

        Args:
            num (int): Max length for packing data.
        """
        try:
            self.pack_file.write("__len__", "{}".format(num).encode("ascii"))
        except ValueError:
            logger.warning(
                f"{self.target_pack_type} is not supported for write length "
                + "mainly due to the string key.",
            )
        self.pack_file.close()

    def _sp_process(self):
        pre_time = time.time()
        idx = 0
        for i in range(self.num_blocks):
            block = self.pack_data(i)
            self._write(i, block)

            if i % 100 == 0:
                cur_time = time.time()
                logger.info(f"time: {cur_time - pre_time}, count: {idx}")
                pre_time = cur_time
            idx = i
        self._write_length(idx + 1)

    def _mp_process(self):
        ctx = multiprocessing.get_context(self.start_method)
        queue = ctx.Queue(self.queue_maxsize)

        rank = int(self.num_blocks / self.num_workers)
        if rank * self.num_workers != self.num_blocks:
            rank += 1
        data_idx = list(range(self.num_blocks))

        read_process = [
            ctx.Process(
                target=_read_worker,
                args=(self, data_idx[i * rank : (i + 1) * rank], queue),
            )
            for i in range(self.num_workers)
        ]
        error_queues = [
            multiprocessing.SimpleQueue() for i in range(self.num_workers)
        ]

        write_process = ctx.Process(target=_write_worker, args=(self, queue))

        for p in read_process:
            p.start()

        error_context = ProcessContext(read_process, error_queues)
        write_process.start()

        error_context.join()

        for p in read_process:
            p.join()

        queue.put(None)
        write_process.join()

        for p in read_process + [write_process]:
            if p.is_alive():
                p.terminate()
