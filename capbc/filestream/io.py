import os
import shutil
import hdfs
from functools import reduce
from contextlib import contextmanager
import tempfile
import sys
import logging
from typing import Tuple, Union, Optional
from .bucket.client import DMP_PREFIX, BucketClient, split_dmp_url
from ..resource_manager import get_current_resource_manager, get_resource


# TODO: refactor using fsspec


__all__ = [
    'open_file', 'reader', 'exists', 'normpath', 'makedirs',
    'delete', 'listdir', 'isdir', "WorkingDir",
]


HDFS_PREFIX = "hdfs://"

HDFS_NODES_DICT = {}

_node2client = dict()

_dmp_client = None
_bucket_client = None


def _get_hdfs_client(namenode):
    global _node2client
    assert namenode in HDFS_NODES_DICT, \
        "Only support hdfs {}, given {}".format(
            HDFS_NODES_DICT.keys(), namenode)
    for node in HDFS_NODES_DICT[namenode]:
        if node not in _node2client:
            try:
                hdfs_client = hdfs.InsecureClient(
                    "http://" + node + ":50070")
                hdfs_client.status('/')
                _node2client[node] = hdfs_client
                return hdfs_client
            except Exception as e:
                continue
        else:
            return _node2client[node]
    raise IOError("HDFS access error. Nodename: {}".format(namenode))


def _get_bucket_client():
    global _bucket_client
    if _bucket_client is None:
        _bucket_client = get_current_resource_manager() and \
            get_resource(BucketClient) or BucketClient()
    return _bucket_client


def _split_path(path) -> Tuple[
        str, str, Optional[Union[hdfs.InsecureClient, BucketClient]]]:
    ''' Split path to 2 parts: head + uri.
    Head contains proto and host(port) for hdfs.
    URI is the unique path in the specified filesystem.

    Parameters
    ----------
    path : string
        file path

    Returns
    -------
    head: string
        the head of path.
    URI: string
        the unique path for the file
    fs_client: hdfs.InsecureClient or DMPClient
        the fs client
    '''

    path_uri = path.strip()
    if isinstance(path_uri, bytes):
        path_uri = path_uri.decode()
    head = ''
    uri = path
    fs_client = None
    if path_uri.startswith('hdfs://'):
        # hdfs
        parts = path_uri[7:].partition('/')
        head = 'hdfs://' + parts[0]
        if parts[1] == '' or parts[2] == '':
            uri = '/'
        else:
            uri = parts[1] + parts[2]
            fs_client = _get_hdfs_client(parts[0])
    elif path_uri.startswith(DMP_PREFIX):
        """
        for dmpv2://matrix/my_tmp,
        result:   head: matrix  uri: /my_tmp
        """
        head, uri = split_dmp_url(path_uri)
        fs_client = _get_bucket_client()
    return head, uri, fs_client


def _hdfs_file_exist(client, path):
    return client.status(path, strict=False) is not None


def open_file(path, *args, **kwargs):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        local_path = path
    elif isinstance(fs_client, BucketClient):
        local_path = fs_client.url_to_local(path)
    elif isinstance(fs_client, hdfs.InsecureClient):
        raise NotImplementedError("Not implemented for hdfs path")
    else:
        raise TypeError(f"Not supported head: {head}")
    return open(local_path, *args, **kwargs)


@contextmanager
def reader(path, encoding=None, delimiter=None,
           buffer_size=128 * 1024 * 1024,
           mode='r'):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        with open(uri, mode, encoding=encoding,
                  newline=delimiter) as f:
            yield f
    elif isinstance(fs_client, hdfs.InsecureClient):
        with fs_client.read(uri, encoding=encoding,
                            delimiter=delimiter,
                            buffer_size=buffer_size) as f:
            yield f
    else:
        raise TypeError("Not supported head: {}".format(head))


def exists(path):
    """ Whether path exists."""
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        return os.path.exists(uri)
    elif isinstance(fs_client, hdfs.InsecureClient):
        return _hdfs_file_exist(fs_client, uri)
    else:
        return _get_bucket_client().exists(path)


def normpath(path):
    """Normalize path, eliminating double slashes, etc.
    """
    if path.startswith('hdfs://'):
        return 'hdfs://' + os.path.normpath(path[7:])
    elif path.startswith(DMP_PREFIX):
        return DMP_PREFIX + os.path.normpath(path[len(DMP_PREFIX):])
    else:
        return os.path.normpath(path)


def makedirs(path, mode=0o777):
    if path[-1] == '/':
        path = path[:-1]
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        if not os.path.exists(uri):
            os.makedirs(uri, mode=mode, exist_ok=True)
    elif isinstance(fs_client, hdfs.InsecureClient):
        # os.makedirs will convert mode to octal representation
        # but hdfs.client.makedirs will not, so we have to convert
        # it here
        fs_client.makedirs(uri, permission=int(oct(mode)[2:]))
    else:
        _get_bucket_client().mkdir(path, recursive=True)


def delete(path, recursive=False):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        if recursive:
            try:
                shutil.rmtree(uri)
                return True
            except Exception:
                return False
        else:
            try:
                os.remove(uri)
                return True
            except Exception:
                return False
    elif isinstance(fs_client, hdfs.InsecureClient):
        try:
            return fs_client.delete(uri, recursive=recursive)
        except hdfs.HdfsError:
            return fs_client.delete(uri, recursive=False)
    else:
        return _get_bucket_client().delete(path, recursive=recursive)


def listdir(path, force_sync=False, force_sync_timeout=3600):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        return os.listdir(uri)
    elif isinstance(fs_client, hdfs.InsecureClient):
        return fs_client.list(uri)
    else:
        return list(_get_bucket_client().ls_iterator(path, detail=False,
                                                     full_path=False,
                                                     force_sync=force_sync,
                                                     force_sync_timeout=force_sync_timeout))  # noqa


def isdir(path: str) -> bool:
    """Similar to os.path.isdir but works for hdfs and gpfs path.
    If path does not exist, this function return False.

    Args:
        path (str): The path

    Raises:
        Exception: If error happens.

    Returns:
        bool: Whether the target path is a directory.
    """
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        return os.path.isdir(uri)
    elif isinstance(fs_client, hdfs.InsecureClient):
        try:
            return fs_client.status(uri)["type"] == "DIRECTORY"
        except hdfs.util.HdfsError as e:
            if e.message.startswith("File does not exist"):
                return False
            else:
                raise e
    else:
        assert isinstance(fs_client, BucketClient)
        try:
            return fs_client.file_info(path).folder
        except FileNotFoundError as e:
            return False


class WorkingDir:
    def __init__(self, target: str):
        self._old = None
        self.target = target

    def __enter__(self):
        if self._old is not None:
            return self
        self._old = os.getcwd()
        os.chdir(self.target)

    def __exit__(self, ptype, value, trace):
        assert self._old is not None
        os.chdir(self._old)
        self._old = None
