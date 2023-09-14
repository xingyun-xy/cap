import logging
import time

from cap.registry import build_from_registry


class _UnsetFlag:
    pass


unset_flag = _UnsetFlag()
logger = logging.getLogger(__name__)


class LmdbDefaultPacker(object):
    """
    Lmdb image record data packer.

    Parameters
    ----------
    cfg : Packing data config.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = cfg.logger
        self._viz_num_show = cfg.get("viz_num_show", -1)
        self._num_workers = cfg.num_workers
        self._data_packer = unset_flag
        self._anno_transformer = unset_flag
        self._anno_packer = unset_flag
        self._viz_fn = unset_flag
        self._viz_dataset = unset_flag
        self._verbose = cfg.get("verbose", False)
        if "anno_ts_fn_config" in cfg:
            self._shuffle_anno = cfg.anno_ts_fn_config.get("shuffle", True)
        else:
            self._shuffle_anno = cfg.get("shuffle", True)
        self._use_roilist = cfg.get("use_roilist", False)

    @property
    def data_packer(self):
        if self._data_packer is unset_flag:
            # pprint.pprint(self.cfg.data_packer)
            self._data_packer = build_from_registry(self.cfg.data_packer)
        return self._data_packer

    @property
    def viz_dataset(self):
        if self._viz_dataset is unset_flag:
            self._viz_dataset = build_from_registry(self.cfg.viz_dataset)
        return self._viz_dataset

    @property
    def viz_fn(self):
        if self._viz_fn is unset_flag:
            self._viz_fn = build_from_registry(self.cfg.viz_fn)
        return self._viz_fn

    def pack_idx(self):
        self.logger.info(
            "~~~Packing index data for task %s~~~" % self.cfg.task_name
        )
        self.data_packer.pack_idx()

    def pack_img(self):
        self.logger.info(
            "~~~Packing image data for task %s~~~" % self.cfg.task_name
        )
        self.logger.info("Begin packing...")
        self.data_packer.pack_img()

    def pack_anno(self):
        verborse_list = []
        self.logger.info(
            "~~~Packing ts anno for task %s~~~" % self.cfg.task_name
        )

        tic = time.time()
        self.data_packer.pack_anno()
        toc = time.time()
        verborse_list.append(("pack ts anno", toc - tic))

        if self._verbose:
            for name, cost_time in verborse_list:
                self.logger.info(name + str(cost_time))

    def visualize(self):
        self.logger.info("Visualizing task %s" % self.cfg.task_name)

        try:
            viz_dataset = self.viz_dataset
        except AttributeError:
            print("no visualizing packed data due to the lack of viz_dataset!")
            return

        for idx, item in enumerate(viz_dataset):
            self.viz_fn(**item)
            if idx >= self._viz_num_show and self._viz_num_show > 0:
                break
