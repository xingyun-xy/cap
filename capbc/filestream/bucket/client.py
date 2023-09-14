""" The client to manipulate files in bucket.

"""  # noqa
import requests
import logging
import os
import subprocess
import yaml
import json
import getpass
import uuid
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from dataclasses_json import DataClassJsonMixin

from capbc.utils import wait_until_finish, call_and_retry, _as_list, deprecated
from capbc.utils.http import _request_stop_cond
from capbc.utils.timeout import timeout_call
from capbc.utils.shell import get_bin_path


__all__ = ['BucketClient', "get_bucket_client"]


logger = logging.getLogger(__name__)

DMP_PREFIX = 'dmpv2://'

GPFS_MKDIR_API = '%s/dmp/v1/buckets/%s/files:mkdir'
GPFS_UPLOAD_FILE_API = '%s/dmpsvc/http-dmp-upload/upload/%s'
GPFS_DOWNLOAD_FILE_API = '%s/dmpsvc/http-dmp-download/download/%s'
GPFS_DELETE_FILE_API = '%s/dmp/v1/buckets/%s/files:batchDelete'
GPFS_FILE_INFO_API = '%s/dmp/v1/buckets/%s/files/detail'
GPFS_EXIST_API = '%s/dmp/v1/buckets/%s/files/isExist'
GPFS_LIST_API = '%s/dmp/v1/buckets/%s/files'
GPFS_FORCE_SYNC_API = '%s/dmp/v1/buckets/%s/files:force_sync'
GPFS_GET_SYNC_TASK_API = '%s/dmp/v1/buckets/%s/files:get-sync-task'
GPFS_RECALL_FROM_TAPE_API = '%s/dmp/v1//buckets/%s/recall'
GPFS_GET_RECALL_FROM_TAPE_TASK_API = '%s/dmp/v1/buckets/tape-task/%s'


B_PER_MB = 2 ** 20


def _parse_mount_root(msg, check_access=True):
    mount_keys = []

    bucket2root = dict()

    for result_i in msg.split('\n')[1:]:
        outputs = []
        for split_i in result_i.split(' '):
            if split_i == '':
                continue
            outputs.append(split_i)
        if outputs:
            outputs = [outputs[0], outputs[-1]]
            for key in mount_keys:
                if key in outputs[0]:
                    bucket = outputs[1].split('/')[-1]
                    if check_access:
                        if os.access(outputs[-1], os.R_OK):
                            bucket2root[bucket] = outputs[-1]
                    else:
                        bucket2root[bucket] = outputs[-1]
                    break

    return bucket2root


def get_bucket_mount_root() -> Dict[str, str]:

    if is_running_on_sda():

        bucket2root = dict()
        prefix = 'bucket_'
        for key_i, val_i in os.environ.items():
            if key_i.startswith(prefix):
                bucket2root[key_i[len(prefix):]] = val_i
        return bucket2root
    else:

        p = subprocess.Popen(
            ['df', '-h', '-t', 'nfs4', '-t', 'fuse.juicefs'], close_fds=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        p.wait()
        results = p.stdout.read().decode()
        p.stdin.close()
        p.stdout.close()

        return _parse_mount_root(results)


@deprecated("Please use `get_bucket_mount_root`.", eos="0.8.0")
def get_gpfs_bucket_mount_root():
    return get_bucket_mount_root()


def is_dmp_url(url: str) -> bool:
    return url.startswith(DMP_PREFIX)


def check_is_dmp_url(url: str):
    if not is_dmp_url(url):
        raise Exception(f'{url} is not a dmp url')


def split_dmp_url(url: str) -> Tuple[str, str]:
    check_is_dmp_url(url)
    dmp_path = url[len(DMP_PREFIX):]
    bucket_name_pos = dmp_path.find('/')
    bucket_name = dmp_path[:bucket_name_pos]
    bucket_file_path = dmp_path[bucket_name_pos:]
    return bucket_name, bucket_file_path


def _download_to_file(response, target_path, with_progress=True):
    chunk_size = 4 * B_PER_MB
    if with_progress:
        pbar = tqdm(desc=target_path, unit='MB')
    else:
        pbar = None
    with open(target_path, 'wb') as fwrite:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fwrite.write(chunk)
            chunk_mb = len(chunk) / float(B_PER_MB)
            if pbar is not None:
                pbar.update(chunk_mb)
    if pbar is not None:
        pbar.close()


@dataclass
class FileInfo(DataClassJsonMixin):
    bucket_name: str
    name: str
    type: str
    size: int
    path: str
    sha256: int
    description: str
    update_at: str
    update_user: str
    folder: bool
    total_file_size: int
    total_file_count: int
    file_id: str
    create_user: str
    # whether data is stored in tape.. It should be bool value
    tape: bool

    def __post_init__(self):
        # hotfix to make tape into bool
        if isinstance(self.tape, int):
            self.tape = True if self.tape == 1 else False



_client = None


def get_bucket_client():
    global _client
    if _client is not None:
        return _client
    _client = BucketClient()
    return _client


class BucketClient(object):
    def __init__(self, token=None, host='',
                 max_retry=5, check_mounted_buckets_visiable=True):
        self._token = token
        self._host = host
        self._max_retry = max_retry

        if token is None:
            self._token = get_user_token()

        if not self._host.startswith("http"):
            self._host = "http://" + self._host

        self._session = requests.Session()
        self.bucket2root = get_bucket_mount_root()

        self._init_hitc(self._token)

        self.check_mounted_buckets_visiable = check_mounted_buckets_visiable
        self.checked_buckets = set()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['bucket2root'] = None
        state['checked_buckets'] = set()
        state['_session'].close()
        state["_session"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        # NOTE: reset this value because the local and remote value
        # is different
        self.bucket2root = get_bucket_mount_root()
        self._init_hitc(self._token)
        self._session = requests.Session()

    def _check_bucket(self, bucket_name):
        """
        Sometimes, bucket may be blocked, for the first time of visiting,
        check whether can be visited or not.
        """
        if bucket_name in self.checked_buckets:
            return
        assert bucket_name in self.bucket2root, f'{bucket_name} does not mount!'  # noqa
        bucket_root = self.bucket2root[bucket_name]
        try:
            timeout_call(120, os.listdir, (bucket_root, ))
        except TimeoutError:
            raise OSError(f'Cannot visit the mounted bucket {bucket_name}, exceed 120 seconds')  # noqa
        self.checked_buckets.add(bucket_name)

    def __del__(self):
        if hasattr(self, '_session') and self._session is not None:
            self._session.close()
            self._session = None

    def _init_hitc(self, token):

        flag, _ = get_bin_path('hitc')
        if not flag:
            message = 'cannot found hitc, ignore dump hitc config'
            logger.warn(message)
            return

        def _impl():
            if os.path.exists(HITC_CONFIG_PATH):
                pre_config = yaml.safe_load(open(HITC_CONFIG_PATH, 'r'))
                if pre_config['token'] == token:
                    return
            subprocess.check_call(['hitc', 'init', '--token', token])

        call_and_retry(
            _impl, catch_exception=(subprocess.CalledProcessError, ),
            retry_interval=(0, 10), max_retry=10)

    def valid_url(self, url: str) -> bool:
        return url.startswith(DMP_PREFIX)

    def mount_buckets(self):
        return list(self.bucket2root.keys())

    def get_mount_root(self, bucket_name) -> str:
        if bucket_name not in self.bucket2root:
            raise FileNotFoundError(f'bucket not mounted: {bucket_name}')
        if self.check_mounted_buckets_visiable:
            self._check_bucket(bucket_name)
        return self.bucket2root[bucket_name]

    def url_to_local(self, url: str, root_only: Optional[bool] = False) -> str:
        bucket_name, bucket_file_path = split_dmp_url(url)
        local_root = self.get_mount_root(bucket_name)
        return local_root if root_only else local_root + bucket_file_path

    def local_to_url(self, path: str) -> str:
        path = os.path.abspath(path)
        for bucket_name in self.bucket2root:
            local_root = self.bucket2root[bucket_name]
            if path.startswith(local_root):
                path = f'{DMP_PREFIX}{bucket_name}{path[len(local_root):]}'
                return path
        else:
            raise FileNotFoundError('Bucket not found for path: %s' % path)

    def mkdir(self, url, recursive=False):
        bucket_name, bucket_file_path = split_dmp_url(url)
        data = {'directories': [bucket_file_path], 'recursive': recursive}
        flag, resp = self._post(
            url=self._get_api('mkdir', bucket_name),
            data=json.dumps(data))
        if flag and not resp.json():
            return True
        else:
            raise OSError(resp.json())

    def exists(self, url: str) -> bool:
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._get(
            url=self._get_api('file_exist', bucket_name),
            params=dict(path=bucket_file_path))
        if flag:
            return resp.content == b'true'
        else:
            raise OSError(resp.text)

    def file_info(self, url: str):
        if not self.exists(url):
            raise FileNotFoundError(f'{url} does not exists')
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._get(
            url=self._get_api('file_info', bucket_name),
            params=dict(path=bucket_file_path))
        if flag:
            ret = FileInfo.from_json(resp.content)
            return ret
        else:
            raise OSError(resp)

    def isfile(self, url: str) -> bool:
        if not self.exists(url):
            return False
        file_info = self.file_info(url)
        return file_info is not None and file_info.folder is False

    def ls_iterator(self, url: str, detail=True, full_path=False,
                    force_sync=False, force_sync_timeout=3600):  # noqa
        if force_sync:
            self.force_sync(url, wait=True, timeout=force_sync_timeout)
        bucket_name, bucket_file_path = split_dmp_url(url)

        if self.isfile(url):
            raise FileNotFoundError("{} is not a folder".format(url))

        if not self.exists(url):
            return iter([])

        def list_file_iter():
            params = {
                'path': bucket_file_path, "penetrate": False,
                "page_token": "page_number=1",
            }
            page_num = 1
            visit_idx = 0
            while True:
                params["page_token"] = "page_number={}".format(page_num)
                flag, resp = self._get(
                    url=self._get_api('list', bucket_name),
                    params=params)
                content = None
                page_num += 1
                if flag:
                    content = resp.json()
                    for f in content["files"]:
                        if not detail:
                            if full_path:
                                f = '{}/{}'.format(url, f['name'])
                            else:
                                f = f['name']
                        visit_idx += 1
                        yield f
                    if visit_idx >= int(content['total_count']):
                        break
                else:
                    break

        return list_file_iter()

    def ls_count(self, url: str):
        bucket_name, bucket_file_path = split_dmp_url(url)
        if not self.exists(url):
            return 0
        if self.isfile(url):
            raise NotADirectoryError("{} is not a folder".format(url))
        params = {
            'path': bucket_file_path,
            "penetrate": False,
            "page_token": "page_number=1",
        }
        flag, resp = self._get(
            url=self._get_api('list', bucket_name),
            params=params)
        if flag:
            content = json.loads(resp.content)
            return int(content["total_count"])
        else:
            return 0

    def download(self, url, target, offset=None, length=None, stream=True,
                 with_progress=True, overwrite=False):
        if os.path.exists(target):
            if overwrite:
                os.remove(target)
            else:
                raise FileExistsError(f'File {target} already exist!')
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._get(
            url=self._get_api('download_file', bucket_name) + bucket_file_path,
            params=dict(offset=offset, length=length), stream=stream)
        if not flag:
            raise OSError(resp)
        else:
            _download_to_file(resp, target, with_progress=with_progress)
            return True

    def upload(self, local_file, url, with_progress=True, timeout=None,
               overwrite=True):

        if not overwrite:
            raise NotImplementedError('overwrite=False is not supported')

        target_url = url
        bucket_name, bucket_file_path = split_dmp_url(url)
        fd, f = os.path.split(bucket_file_path)
        url = self._get_api('upload_file', bucket_name) + fd

        from requests_toolbelt.multipart.encoder import (
            MultipartEncoder, MultipartEncoderMonitor)
        from capbc.filestream.io import reader

        def create_callback(encoder: MultipartEncoder, bar_obj: tqdm):
            encoder_len = encoder.len
            bar_obj.total = encoder_len/float(B_PER_MB)

            def callback(monitor):
                read_size = monitor.bytes_read/float(B_PER_MB)
                bar_obj.update(read_size - bar_obj.n)
            return callback

        def _impl():

            with reader(local_file, mode='rb') as fin:
                payload = {'file': (os.path.basename(f),
                                    fin, 'application/octet-stream')}
                monitor = MultipartEncoder(payload)
                bar_object = None
                if with_progress:
                    bar_format = "{desc}:{percentage:.2f}%|{bar}|" + \
                        "{n:.3f}/{total:.3f}" + \
                        "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                    bar_object = tqdm(
                        desc=local_file, unit='MB', total=0,
                        bar_format=bar_format,
                    )
                    callback = create_callback(monitor, bar_object)
                    monitor = MultipartEncoderMonitor(monitor, callback)
                flag, resp = self._post(url=url, data=monitor,
                                        content_type=monitor.content_type,
                                        allow_redirects=True, max_retry=0)
                if bar_object is not None:
                    bar_object.close()

            if flag:
                return True
            else:
                raise OSError(resp)

        flag, msg = call_and_retry(
            timeout_call, args=(timeout, _impl),
            kwargs=dict(message='Upload {} to {} exceed {}s'.format(local_file, url, timeout)),  # noqa
            max_retry=self._max_retry)
        if flag:
            return True
        raise OSError(msg)

    def delete(self, url, recursive=False):
        if not self.exists(url):
            return True
        if not self.isfile(url) and not recursive:
            msg = "To delete a folder, you need to set recursive flag"
            raise IsADirectoryError(msg)
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._post(
            url=self._get_api('delete_file', bucket_name),
            data=json.dumps({'files': [bucket_file_path],
                             'recursive': recursive}))
        if flag and not resp.json():
            return True
        else:
            raise OSError(resp)

    def force_sync(self, url, wait=True, timeout=3600):
        bucket_name, bucket_file_path = split_dmp_url(url)

        flag, resp = self._post(
            url=self._get_api('force_sync', bucket_name),
            data=json.dumps(dict(path=bucket_file_path)))

        if flag:
            sync_task_id = resp.json()['id']
        else:
            msg = resp.json()

            # compatible with new API
            if 'already exist' in msg['error']:
                sync_task_id = msg['error'].split(' ')[2]
            else:
                raise RuntimeError(f'Sync failed, msg = {msg}')

        if wait:

            def query_func():
                status = self.get_sync_task_status(bucket_name, sync_task_id)  # noqa
                return status in ['FAILED', 'COMPLETED']

            wait_until_finish(query_func, timeout=timeout,
                              wait_what=f'GPFS force_sync {url}, task_id {sync_task_id}')  # noqa
            return self.get_sync_task_status(bucket_name, sync_task_id) == 'COMPLETED'  # noqa
        else:
            return sync_task_id

    def get_sync_task_status(self, bucket_name, task_id):
        flag, resp = self._get(
            url=self._get_api('get_sync_task', bucket_name),
            params=dict(task_id=task_id))
        if flag:
            return resp.json()['status']
        else:
            raise RuntimeError(f'Get sync task status failed, msg = {resp.json()}')  # noqa

    def recall_from_tape(self, paths, priority=3, wait=False,
                         timeout=3600 * 24):
        """
        Recall files from tape

        Parameters
        ----------
        paths : list/tuple of str
            All paths should be in the same bucket.
        priority : int, optional
            Task priority, possible values are {1, 2, 3, 4, 5}, by default 3
        wait : bool, optional
            Whether wait until recall task finished, by default False
        timeout : float, optional
            Maximuum waiting time, by default 3600 * 24 (24 hours).
        """
        paths = _as_list(paths)
        assert all([self.valid_url(path_i) for path_i in paths]), \
            f'All paths should starts with {DMP_PREFIX}, but get {paths}'

        split_bucket_name = None
        file_paths = []
        for path_i in _as_list(paths):
            bucket_name, bucket_file_path = split_dmp_url(path_i)
            if split_bucket_name is None:
                split_bucket_name = bucket_name
            else:
                assert bucket_name == split_bucket_name, \
                    f'all paths should in the same bucket, but get {bucket_name} vs. {split_bucket_name}'  # noqa
            file_paths.append(bucket_file_path)

        assert file_paths, f'paths = {paths} is not allowed to be empty'

        flag, resp = self._post(
            url=self._get_api('recall_from_tape', split_bucket_name),
            data=json.dumps(dict(
                file_list=file_paths,
                priority=priority,
            ))
        )
        ret = resp.json()

        if flag:
            task_id = ret['content']['job_id']

            if task_id == 0:
                raise RuntimeError(f'Recall from tape failed, msg = {ret}')

            if wait:

                def _query_func():
                    status = self.get_recall_from_tape_task_status(task_id=task_id)  # noqa
                    return status in ['completed', 'failed', 'cancaled']

                wait_until_finish(_query_func, timeout=timeout)
                return self.get_recall_from_tape_task_status(task_id=task_id) == 'completed'  # noqa

            else:
                return task_id
        else:
            raise RuntimeError(f'Recall from tape failed, msg = {ret}')

    def get_recall_from_tape_task_status(self, task_id):
        """
        Get recall from tape task status.

        Parameters
        ----------
        task_id : int
            Task id
        """
        flag, resp = self._get(
            url=self._get_api('get_recall_from_tape_task', task_id),
            params=None)

        if flag:
            ret = resp.json()['content']
            assert len(ret) == 1, ret
            return ret[0]['status']
        else:
            raise RuntimeError(f'Geet recall from tape task status failed, msg = {resp.json()}')  # noqa

    # ----- Internal functions -----

    def _get(self, url, params, allow_redirects=False,
             content_type='application/json',
             stop_cond=_request_stop_cond, max_retry=None,
             **kwargs):
        headers = {
            'Content-Type': content_type,
            'X-Forwarded-User': self._token}
        max_retry = self._max_retry if max_retry is None else max_retry
        flag, resp = call_and_retry(
            self._session.get,
            kwargs=dict(
                url=url,
                params=params,
                headers=headers,
                allow_redirects=allow_redirects,
                **kwargs
            ),
            max_retry=max_retry,
            stop_cond=stop_cond,
        )
        return flag, resp

    def _post(self, url, data, allow_redirects=False,
              content_type='application/json',
              stop_cond=_request_stop_cond, max_retry=None,
              **kwargs):
        headers = {
            'Content-Type': content_type,
            'X-Forwarded-User': self._token}
        max_retry = self._max_retry if max_retry is None else max_retry
        flag, resp = call_and_retry(
            self._session.post,
            kwargs=dict(
                url=url,
                data=data,
                headers=headers,
                allow_redirects=allow_redirects,
                **kwargs
            ),
            max_retry=max_retry,
            stop_cond=stop_cond,
        )
        return flag, resp

    def _get_api(self, api_name, bucket_name):
        if api_name == 'mkdir':
            return GPFS_MKDIR_API % (self._host, bucket_name)
        elif api_name == 'upload_file':
            return GPFS_UPLOAD_FILE_API % (self._host, bucket_name)
        elif api_name == 'download_file':
            return GPFS_DOWNLOAD_FILE_API % (self._host, bucket_name)
        elif api_name == 'delete_file':
            return GPFS_DELETE_FILE_API % (self._host, bucket_name)
        elif api_name == 'file_info':
            return GPFS_FILE_INFO_API % (self._host, bucket_name)
        elif api_name == 'file_exist':
            return GPFS_EXIST_API % (self._host, bucket_name)
        elif api_name == 'list':
            return GPFS_LIST_API % (self._host, bucket_name)
        elif api_name == 'force_sync':
            return GPFS_FORCE_SYNC_API % (self._host, bucket_name)
        elif api_name == 'get_sync_task':
            return GPFS_GET_SYNC_TASK_API % (self._host, bucket_name)
        elif api_name == 'recall_from_tape':
            return GPFS_RECALL_FROM_TAPE_API % (self._host, bucket_name)
        elif api_name == 'get_recall_from_tape_task':
            return GPFS_GET_RECALL_FROM_TAPE_TASK_API % (self._host, bucket_name)  # noqa
        else:
            raise ValueError(f'Unsupported api {api_name}')
