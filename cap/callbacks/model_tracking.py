import logging

from cap.core.event import EventStorage
from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import _as_list
from cap.utils.distributed import get_dist_info, rank_zero_only
from .callbacks import CallbackMixin

logger = logging.getLogger(__name__)


__all__ = ["ModelTracking"]


class MetricsStorageHandler(object):
    """The metrics handler, using event storage to transfer between\
    different callbacks."""

    metrics_key = "_metrics_pairs"
    syn_key = "_metrics_syn"
    ack_key = "_metrics_ack"

    @classmethod
    def consumer_syn(cls, storage: EventStorage):
        """To tell storage that consumer is ready."""
        storage.put(cls.syn_key, True)

    @classmethod
    def consumer_syned(cls, storage: EventStorage):
        """To check if consumer is ready."""
        try:
            return storage.get(cls.syn_key)[0] is True
        except KeyError:
            pass

        return False

    @classmethod
    def producer_ack(cls, storage: EventStorage):
        """To tell storage that producer is ready."""
        storage.put(cls.ack_key, True)

    @classmethod
    def producer_acked(cls, storage: EventStorage):
        """To check if producer is ready."""
        try:
            return storage.get(cls.ack_key)[0] is True
        except KeyError:
            pass

        return False

    @classmethod
    def simple_produce(cls, storage: EventStorage, train_metrics: list):
        """To produce from train metrics to event storage.

        If the consumer is not ready, produce will stop.
        """
        if cls.consumer_syned(storage):
            for metric in train_metrics:
                name, value = metric.get()
                for k, v in zip(_as_list(name), _as_list(value)):
                    if isinstance(v, (int, float, str)):
                        storage.put(cls.metrics_key, (k, v))
                    else:
                        # skip the k-v which value is not int、float、str
                        pass

    @classmethod
    def simple_consume(cls, storage: EventStorage):
        """To consume from event storage."""
        if cls.producer_acked(storage):
            try:
                metrics_list = storage.get(cls.metrics_key)
                storage.clear_key(cls.metrics_key)
                return metrics_list
            except KeyError:
                pass
        return []


class ModelStorageHandler(object):
    """The model handler, using event storage to transfer between\
    different callbacks."""

    model_key = "_model_dict"

    @classmethod
    def produce(cls, storage: EventStorage, model_dict: dict):
        """To produce to event storage."""
        storage.put(cls.model_key, model_dict)

    @classmethod
    def consume(cls, storage: EventStorage):
        """To consume from event storage."""
        try:
            model_list = storage.get(cls.model_key)
            storage.clear_key(cls.model_key)
            return model_list
        except KeyError:
            pass
        return []


@OBJECT_REGISTRY.register
class ModelTracking(CallbackMixin):
    """Make model training and prediction trackable.

    Args:
        job_type (str): to track a train or a prediction job.
        update_freq (int): the step update frequency. bigger than 0.
        model_name (str): the model name.
        model_version (str): the model version which track infos link to.
    """

    def __init__(
        self,
        job_type: str = "train",
        update_freq: int = 0,
        model_name: str = None,
        model_version: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._client = None
        self.job_type = job_type
        self.update_freq = update_freq
        self.model_name = model_name
        self.model_version = model_version

    @property
    def client(self):
        if self._client is None:
            self._client = get_sd_client()
            if self.job_type is not None:
                self._client.tracker.set_run_info(
                    job_type=self.job_type,
                )
            logger.info(
                "client initialized with job type[%s], run info[%s]"
                % (self.job_type, self._client.tracker.run_info)
            )
        return self._client

    @rank_zero_only
    def on_loop_begin(self, storage: EventStorage, **kwargs):
        logger.debug("begin consumer syn")
        MetricsStorageHandler.consumer_syn(storage)

    @rank_zero_only
    def on_step_end(
        self,
        train_metrics,
        storage,
        epoch_id=None,
        step_id=None,
        global_step_id=None,
        **kwargs,
    ):
        rank, world_size = get_dist_info()

        # handle metrics from event storage
        logger.debug("check producer acked or not")
        if MetricsStorageHandler.producer_acked(storage):
            logger.debug("producer acked, start to consume from storage")
            metrics_list = MetricsStorageHandler.simple_consume(storage)
            logger.debug("consume metrics list: %s" % metrics_list)
            for (k, v) in metrics_list:
                self.client.tracker.log_metrics(
                    key=k,
                    value=v,
                    step=step_id,
                    epoch_id=epoch_id,
                    global_step_id=global_step_id,
                    rank=rank,
                    world_size=world_size,
                )

        # handle train metrics
        if self.update_freq > 0 and (step_id + 1) % self.update_freq == 0:
            for metric in train_metrics:
                name, value = metric.get()
                if name and value:
                    logger.debug(
                        "freq reached, metric name[%s] value[%s]"
                        % (name, value)
                    )
                    for k, v in zip(_as_list(name), _as_list(value)):
                        if isinstance(v, (int, float, str)):
                            self.client.tracker.log_metrics(
                                key=k,
                                value=v,
                                step=step_id,
                                epoch_id=epoch_id,
                                global_step_id=global_step_id,
                                rank=rank,
                                world_size=world_size,
                            )
                        else:
                            # skip the k-v which value is not int nor float
                            pass

    @rank_zero_only
    def on_loop_end(self, storage: EventStorage, **kwargs):
        if self.model_name and self.model_version:
            logger.debug(
                "link %s to setted model %s,%s"
                % (
                    self.client.tracker.run_info,
                    self.model_name,
                    self.model_version,
                )
            )
            self.client.tracker.link_to_model(
                self.client.model.finditem(self.model_name, self.model_version)
            )

        model_list = ModelStorageHandler.consume(storage)
        for model_dict in model_list:
            model_name = model_dict.get("model_name")
            model_version = model_dict.get("model_version")
            logger.debug(
                "link %s to consumed model %s,%s"
                % (
                    self.client.tracker.run_info,
                    model_name,
                    model_version,
                )
            )
            self.client.tracker.link_to_model(
                self.client.model.finditem(model_name, model_version)
            )
        # TODO(): Add prediction job output track.

        self.client.tracker.wait_task_finish()
