import time
import json
import logging
import requests
from .utils import call_and_retry


__all__ = ['get', 'post']


logger = logging.getLogger(__name__)


def _request_stop_cond(resp):
    return resp.status_code == requests.codes.ok


def get(url, headers, params, session=None,
        max_retry=0, retry_interval=1, stop_cond=None,
        **kwargs):

    flag, resp = call_and_retry(
        requests.get if session is None else session.get,
        kwargs=dict(
            url=url,
            params=params,
            headers=headers,
            **kwargs),
        stop_cond=_request_stop_cond if stop_cond is None else stop_cond,
        max_retry=max_retry,
        retry_interval=retry_interval)
    return flag, resp


def post(url, data, headers, session=None,
         max_retry=0, retry_interval=1, stop_cond=None,
         **kwargs):

    flag, resp = call_and_retry(
        requests.post if session is None else session.post,
        kwargs=dict(
            url=url,
            data=data,
            headers=headers,
            **kwargs),
        stop_cond=_request_stop_cond if stop_cond is None else stop_cond,
        max_retry=max_retry,
        retry_interval=retry_interval)
    return flag, resp
