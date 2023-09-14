# Copyright (c) Changan Auto. All rights reserved.

import logging
import time
from functools import partial
from typing import Optional

import numpy as np

from cap.callbacks.metric_updater import MetricUpdater
from cap.callbacks.monitor import StatsMonitor
from cap.data.dataloaders import MultitaskLoader

__all__ = ["ModelTrainingPerf"]

logger = logging.getLogger(__name__)


def fit(trainer, iter_nums, frequent, batch_size):
    """Perf model training, similar to Trainer.fit."""
    self = trainer
    btic = time.time()
    global_step_id = 0
    batch = None
    speed_list, cost_list = [], []

    self.model.train()
    self.on_loop_begin(
        model=self.model, optimizer=self.optimizer, trainer=self
    )

    for epoch_id in range(self.start_epoch, self.num_epochs):
        self.model.train()
        self.on_epoch_begin(epoch_id=epoch_id, optimizer=self.optimizer)

        while True:
            step_id = global_step_id

            if batch is None:
                batch = next(iter(self.data_loader))

            self.on_step_begin(
                optimizer=self.optimizer,
                data_loader=self.data_loader,
                start_epoch=self.start_epoch,
            )

            self.batch_processor(
                batch,
                self.model,
                self.device,
                optimizer=self.optimizer,
                batch_begin_callback=partial(
                    self.on_batch_begin,
                    global_step_id=global_step_id,
                    step_id=step_id,
                ),
                batch_end_callback=partial(
                    self.on_batch_end,
                    global_step_id=global_step_id,
                    step_id=step_id,
                ),
            )

            self.on_step_end(
                epoch_id=epoch_id,
                step_id=step_id,
                data_loader=self.data_loader,
                num_epochs=self.num_epochs,
                model=self.model,
                optimizer=self.optimizer,
                global_step_id=global_step_id,
            )

            if (global_step_id + 1) % frequent == 0:
                cost = time.time() - btic
                speed = frequent * batch_size / cost
                s = f"Batch[{global_step_id}] Speed: {speed:.2f} samples/sec | Cost: {cost:.3f} sec"  # noqa
                logger.info(s)
                speed_list.append(speed)
                cost_list.append(cost)
                btic = time.time()

            global_step_id += 1
            if global_step_id >= iter_nums:
                s = f"Average Batch Speed: {np.mean(speed_list):.2f} samples/sec | Cost: {np.mean(cost_list):.2f} sec"  # noqa
                logger.info(s)
                return

        self.on_epoch_end(
            epoch_id=epoch_id,
            model=self.model,
            optimizer=self.optimizer,
            num_epochs=self.num_epochs,
            device=self.device,
        )

        if global_step_id >= iter_nums:
            break

    self.on_loop_end(
        model=self.model,
        optimizer=self.optimizer,
        num_epochs=self.num_epochs,
        device=self.device,
    )


class ModelTrainingPerf:
    """
    Model training speed perf plugin, used to test model training speed.

    Including forward/backward/step, without loading data.

    Args:
        trainer: trainer instance.
        batch_size: total batch size of one iter from dataloader.
        iter_nums: perf iteration nums.
        frequent: log frequent.
        show_loss: show loss metric.

    """

    def __init__(
        self,
        trainer,
        batch_size: Optional[int] = None,
        iter_nums: Optional[int] = 1000,
        frequent: Optional[int] = 1,
        show_loss: Optional[bool] = False,
    ):
        self.trainer = trainer
        self.iter_nums = iter_nums
        self.frequent = frequent

        # TODO(min.du, 0.1): put batch_size to Trainer #
        dataloader = trainer.data_loader
        if batch_size is not None:
            self.batch_size = batch_size
        elif hasattr(dataloader, "batch_size"):
            self.batch_size = dataloader.batch_size
        elif isinstance(dataloader, MultitaskLoader):
            loaders = dataloader.loaders
            if isinstance(loaders, dict):
                self.batch_size = sum([loaders[k].batch_size for k in loaders])
            elif isinstance(loaders, list):
                self.batch_size = sum([item.batch_size for item in loaders])
            else:
                raise ValueError
        else:
            raise NotImplementedError(f"{type(loaders)} not supported")

        # don't show loss
        if show_loss:
            # change log_interval to frequent
            pass
        else:
            new_callbacks = []
            for cb in self.trainer.callbacks:
                if not isinstance(cb, (StatsMonitor, MetricUpdater)):
                    new_callbacks.append(cb)
            self.trainer.callbacks = new_callbacks

    def run(self) -> None:
        fit(self.trainer, self.iter_nums, self.frequent, self.batch_size)
