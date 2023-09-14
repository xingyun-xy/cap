import glob
import json
import logging
import os
import re
import sys
from importlib import import_module

import numpy as np
import yaml
from easydict import EasyDict
from capbc.utils import _as_list

logger = logging.getLogger(__name__)


def get_pack_list(anno_path, camera_names):
    """Get image and anno list.

    Parameters
    ----------
    anno_path: str.
        Annotation path generated by auto_label, coco format.
    camera_names: list
        List of camera names.

    Return
    ------
    Pack list.
        List of (image, anno)
    """

    anno_files = glob.glob(os.path.join(anno_path, "*.json"))
    with open(anno_files[0], "r") as f:
        json_anno = json.load(f)
    images, annotations = json_anno["images"], json_anno["annotations"]

    imgid2anno = {}
    for im_info in images:
        for camera_name in camera_names:
            if camera_name in im_info["image_key"]:
                imgid2anno[im_info["id"]] = {
                    "meta": im_info,
                    "objects": [],
                    "image_key": os.path.basename(im_info["image_source"]),
                }

    for anno in annotations:
        img_id = anno["image_id"]
        if img_id in imgid2anno:
            imgid2anno[img_id]["objects"] += [anno]

    pack_list = [
        (os.path.join(anno_path, "data"), anno) for anno in imgid2anno.values()
    ]

    logger.info("Found %d images" % len(pack_list))

    return pack_list


def ls_img_folder_and_anno_file(
    root_path, anno_ext=".json", recursive=True, excluded_annos_pattern=None
):
    """
    List image folders and their annotation files recursively.

    Data should be organize in:

    .. code-block:: none

        root_path[0]/
            image_folder1
            image_folder1.json
            image_folder2
            image_folder2.json
        root_path[1]/
            image_folder3
            image_folder3.json
            ...

    annotation files in ``excluded_annos`` will be ignored.

    Parameters
    ----------
    root_path : list of str, or str
        Root paths for listing. [root_path1, root_path2...].
    anno_ext : str
        The extension of annotation files.
    recursive : bool
        Whether to list recursively.
    excluded_annos : list/tuple of str, or None
        Paths of excluded annotation files.

    Returns
    -------
    folder_anno_pair_list : list of tuples.
        Return list of (image_folder, anno_file) pairs.

    Examples
    --------
    ls_img_folder_and_anno_file(['root_path1', 'root_path2', ...],
         anno_ext='.json', recursive=True,
         excluded_annos=['/path/to/excluded_anno1.json', ...]
    """

    root_path = _as_list(root_path)
    if excluded_annos_pattern is not None:
        excluded_annos_pattern = re.compile(excluded_annos_pattern)

    if not anno_ext.startswith("."):
        anno_ext = "." + anno_ext
    image_folder_list = []
    anno_file_list = []

    def _search_one_path(one_path):
        one_path = os.path.expanduser(one_path)
        anno_files = glob.glob(
            one_path + "/**/*" + anno_ext, recursive=recursive
        )
        for anno_file in anno_files:
            # check abs path
            if excluded_annos_pattern is not None:
                if excluded_annos_pattern.match(anno_file):
                    logger.info("ignore annotation %s" % anno_file)
                    continue
            img_folder = anno_file.replace(anno_ext, "")
            if not os.path.exists(img_folder) or not os.path.exists(anno_file):
                continue
            image_folder_list.append(img_folder)
            anno_file_list.append(anno_file)

    for one_path in root_path:
        _search_one_path(one_path)

    folder_anno_pair_list = list(zip(image_folder_list, anno_file_list))
    return folder_anno_pair_list


def get_colors_and_class_names_for_lane_parsing(color_map):
    """
    Get class name list and the corresponding color list for lane parsing task.

    Parameters
    ----------
    color_map: dict
        The segmentation color map.
    """
    colors, clsnames = get_colors_and_class_names_for_parsing(color_map)
    colors[255, :, :] = [255, 0, 0]  # lane parsing default class

    return colors, clsnames


def get_colors_and_class_names_for_parsing(color_map):
    """Get class name list and the corresponding color list.

    Parameters
    ----------
    color_map: dict
        The segmentation color map.
    """
    colors = np.zeros((256, 1, 3), dtype="uint8")
    clsnames = []
    for i, (class_name, color) in enumerate(color_map.items()):
        colors[i, :, :] = color
        clsnames.append(class_name)

    return colors, clsnames


def _check_path_exists(path):
    fn = os.path.exists
    assert fn(path), "%s does not exists" % path


def load_pyfile(filename, allow_unsafe=False):
    _check_path_exists(filename)
    module_name = os.path.basename(filename)[:-3]
    config_dir = os.path.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith("__")
    }
    sys.modules.pop(module_name)
    return cfg


def load_json(filename, allow_unsafe=False):
    _check_path_exists(filename)
    with open(filename, "r") as f:
        cfg = json.load(f)
    return cfg


def load_yaml(filename, allow_unsafe=False):
    _check_path_exists(filename)
    with open(filename, "r") as f:
        if allow_unsafe:
            cfg = yaml.unsafe_load(f)
        else:
            cfg = yaml.safe_load(f)
    return cfg


ext_to_load_fn_map = {
    ".py": load_pyfile,
    ".json": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
}


def set_pack_env(
    num_worker=16,
    allow_missing=False,
    skip_invalid=False,
    overwrite=False,
    verbose=False,
    pipeline_test=False,
):
    """
    Set environment for auto matrix data packer tools.

    Parameters
    ----------
    num_worker : int, optional
        Parallel numbers, by default 16
    allow_missing : bool, optional
        Whether allowing image provided in annotations by does not exists,
        by default False
    skip_invalid : bool, optional
        Whether skip invalid images, e.g., broken image, by default False
    overwrite : bool, optional
        Whether to overwrite existing records, by default False
    verbose : bool, optional
        Whether to verbose, by default False
    pipeline_test : bool, optional
        Whether in pipeline test mode, by default False
    """

    os.environ.update(
        {
            "PACK_NUM_WORKER": str(num_worker),
            "PACK_ALLOW_MISSING": str(allow_missing),
            "PACK_SKIP_INVALID": str(skip_invalid),
            "PACK_OVERWRITE": str(overwrite),
            "PACK_VERBOSE": str(verbose),
            "PACK_PIPELINE_TEST": str(pipeline_test),
        }
    )


def str_to_bool(s):
    s = s.lower()
    if s in ["true", "t", "1"]:
        return True
    elif s in ["false", "f", "0"]:
        return False
    else:
        raise ValueError("Invalid value %s" % s)


def get_default_pack_args_from_environment(
    default_num_worker=16,
    default_allow_missing="False",
    default_skip_invalid="False",
    default_overwrite="False",
    default_verbose="False",
):
    """
    Get arguments from auto matrix data packer tools environment.

    Parameters
    ----------
    **kwargs :
        Please see :py:func:`set_pack_env`
    """
    num_worker = int(os.environ.get("PACK_NUM_WORKER", default_num_worker))

    allow_missing = str_to_bool(
        os.environ.get("PACK_ALLOW_MISSING", default_allow_missing)
    )

    skip_invalid = str_to_bool(
        os.environ.get("PACK_SKIP_INVALID", default_skip_invalid)
    )

    overwrite = str_to_bool(
        os.environ.get("PACK_OVERWRITE", default_overwrite)
    )

    verbose = str_to_bool(os.environ.get("PACK_VERBOSE", default_verbose))

    pipeline_test = str_to_bool(os.environ.get("PACK_PIPELINE_TEST", "False"))

    return EasyDict(
        {
            "num_worker": num_worker,
            "allow_missing": allow_missing,
            "skip_invalid": skip_invalid,
            "overwrite": overwrite,
            "verbose": verbose,
            "pipeline_test": pipeline_test,
        }
    )


def init_logger(
    log_file,
    logger_name=None,
    rank=0,
    level=logging.INFO,
    overwrite=False,
    stream=sys.stderr,
    log_filepath=False,
    encoding=None,
):
    """Initialize a logger. logger will have the formatter like.

    .. code-block:: none

        '%(asctime)-15s %(levelname)s Node[' + str(rank) + '] %(message)s'

    Parameters
    ----------
    log_file: str
        Open the specified file and use it as the stream for logging.
    logger_name: str
        The logger name.
    rank: int
        The node rank, used for distributed training.
    level: int
        Logging level.
    overwrite: bool
        Whether to overwrite the logging file if it is already exists.
    stream: stream
        Logging stream, can be ``sys.stderr`` or ``sys.stdout``.
    encoding: str or None
        Encoding when open log_file.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir != "":
        os.makedirs(log_dir, exist_ok=True)
    head = "%(asctime)-15s %(levelname)s "
    if log_filepath:
        head += "%(filename)s:L%(lineno)d "
    head += "Node[" + str(rank) + "] %(message)s"
    if rank != 0:
        log_file += "-rank%d" % rank
    if os.path.exists(log_file) and overwrite:
        os.remove(log_file)
    logger = logging.getLogger(logger_name)
    if not logger.handlers:  # duplicate handlers will cause duplicate outputs
        formatter = logging.Formatter(head)
        file_handler = logging.FileHandler(log_file, encoding=encoding)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.setLevel(level)
    return logger
