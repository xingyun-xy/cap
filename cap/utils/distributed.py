# Copyright (c) Changan Auto. All rights reserved.

import logging
import random
import socket
from functools import wraps
from typing import Any, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

__all__ = [
    "find_free_port",
    "get_dist_info",
    "rank_zero_only",
    "get_device_count",
    "get_global_process_group",
    "set_local_process_group",
    "get_local_process_group",
    "split_process_group_by_host",
]

logger = logging.getLogger(__name__)

_GLOBAL_PROCESS_GROUP = None
_LOCAL_PROCESS_GROUP = _GLOBAL_PROCESS_GROUP
_HOST_PROCESS_GROUP_CACHED = {}


def find_free_port(start_port: int = 10001, end_port: int = 19999) -> int:
    """Find free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_port = -1
    max_retry = 300
    for _i in range(0, max_retry):
        port = random.randint(start_port, end_port)
        try:
            sock.bind(("", port))
            free_port = port
            sock.close()
            break
        except socket.error:
            continue
    if free_port == -1:
        raise Exception("can not found a free port")
    return free_port


def rank_zero_only(fn):
    """Migrated from pytorch-lightning."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def dist_initialized():
    initialized = False
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
    return initialized


def get_dist_info(process_group=None) -> Tuple[int, int]:
    """Get distributed information from process group."""
    if dist_initialized():
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_device_count(process_group=None):
    """Return the number of GPUs available."""
    initialized = dist_initialized()
    if initialized:
        _, world_size = get_dist_info(process_group)
        return world_size
    else:
        return torch.cuda.device_count()


def create_process_group(rank_list: Tuple[str]) -> Any:
    """Create a new process group by a list of rank id."""
    group = None
    if dist_initialized():
        group = dist.new_group(rank_list)
    return group


def get_global_process_group():
    global _GLOBAL_PROCESS_GROUP
    return _GLOBAL_PROCESS_GROUP


def set_local_process_group(process_group):
    global _LOCAL_PROCESS_GROUP
    _LOCAL_PROCESS_GROUP = process_group


def get_local_process_group():
    global _LOCAL_PROCESS_GROUP
    return _LOCAL_PROCESS_GROUP


def split_process_group_by_host(
    process_group: ProcessGroup = None,
) -> Tuple[ProcessGroup, bool]:
    """Get process group that only contains localhost rank within process group.

    Args:
        process_group: a process_group which contains local rank.
    """
    if not dist_initialized():
        return process_group, False

    global _HOST_PROCESS_GROUP_CACHED
    if process_group in _HOST_PROCESS_GROUP_CACHED:
        # use cached result for process group
        return _HOST_PROCESS_GROUP_CACHED[process_group], True

    # first get host name/ipaddr for current rank
    try:
        hostid = socket.gethostname()
        if hostid is None:
            hostid = socket.getfqdn()
            if hostid is None:
                # get host failed, fallback to origin process group
                raise Exception("get host name failed")
            if hostid == "localhost":
                hostid = socket.gethostbyname(hostid)
                if hostid == "0.0.0.0":
                    raise Exception("get host name failed")
    except Exception:
        return process_group, False

    # 1. first split process group by host
    current_rank, _ = get_dist_info(None)
    _, local_world_size = get_dist_info(process_group)
    # get all rank and hostid within process_group
    local_data = [current_rank, hostid]
    glob_data = [None for _ in range(local_world_size)]
    all_gather_object(glob_data, local_data, process_group)

    # aggregate ranks by hostid
    glob_host_dict = {}
    for gd in glob_data:
        r, host = gd  # gd is [rank, hostid]
        if host not in glob_host_dict:
            glob_host_dict[host] = [r]
        else:
            glob_host_dict[host].append(r)

    # new group ranks is used to create new groups
    # it is [[r1,r2],[r3,r4]] format.
    # each element will be used to create group
    new_group_ranks = list(glob_host_dict.values())
    if len(new_group_ranks) == 1:
        # the process group all in same host
        # so not need to split
        new_group_ranks = []

    # 2. if process group is not global group, then exchange all split info
    if process_group is not None:
        # exchange group info when process group is not default
        _, world_size = get_dist_info(None)
        glob_data = [None for _ in range(world_size)]
        all_gather_object(glob_data, new_group_ranks, None)
        new_group_ranks = []
        for d in glob_data:
            # d is list of rank-list that on same host
            new_group_ranks.extend(d)

    if len(new_group_ranks) == 0:
        # all process group on one host, not need to create new group
        logger.info(
            f"rank {current_rank} same host {hostid} not need to split"
        )
        return process_group, True

    # create new groups
    result_pg = process_group
    for ranks in new_group_ranks:
        npg = create_process_group(ranks)
        if current_rank in ranks:
            logger.info(f"host: {hostid} create new group for ranks: {ranks}")
            result_pg = npg
    _HOST_PROCESS_GROUP_CACHED[process_group] = result_pg
    return result_pg, True


def all_gather_object(obj_list: List[Any], obj: Any, group=None):
    """Gather object from every ranks in group."""
    if dist_initialized():
        dist.all_gather_object(obj_list, obj, group)
    else:
        assert isinstance(obj_list, list) and len(obj_list) == 1
        obj_list[0] = obj


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device.

    Use 'cpu' for CPU to avoid a deprecation warning.

    When torch version <1.9, `torch.nn.parallel.scatter_gather.gather` does not
    support `namedtuple` outputs, but this one does.
    """
    from cap.utils.apply_func import is_namedtuple

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)  # noqa: F821
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError("All dicts must have the same number of keys")
            return type(out)(
                ((k, gather_map([d[k] for d in outputs])) for k in out)
            )
        if is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


def get_global_out(output):
    global_rank, global_world_size = get_dist_info()
    global_output = [None for _ in range(global_world_size)]
    all_gather_object(global_output, output)
    return global_rank, global_output
