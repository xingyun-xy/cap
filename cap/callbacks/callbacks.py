# Copyright (c) Changan Auto. All rights reserved.

from abc import ABC

__all__ = ["CallbackMixin"]


class CallbackMixin(ABC):
    """Callback interface class."""

    def on_loop_begin(self, **kwargs):
        # Note: Not support *args, because different callback may have
        # different positional arguments, they can ONLY point out necessary
        # arguments, e.g.
        # on_loop_begin(self, model, optimizer, data_loader, **kwargs)
        # on_loop_begin(self, data_loader, **kwargs)
        pass

    def on_loop_end(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_step_begin(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        """There may be multiple batches in a multitask training step."""

    def on_batch_end(self, **kwargs):
        """There may be multiple batches in a multitask training step."""

    def on_backward_begin(self, **kwargs):
        pass

    def on_backward_end(self, **kwargs):
        pass

    def on_optimizer_step_begin(self, **kwargs):
        pass
