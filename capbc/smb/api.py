from abc import ABCMeta, abstractclassmethod, abstractmethod
from requests.utils import dict_from_cookiejar
import fsspec
from .config import Config

__all__ = ['SMBClient']


class SMBClient():
    """[SMB Client based on fsspec]
    """
    def check(self):
        pass  # TODO: find some way to check login

    def __init__(self, username, password, host, logging_level='INFO'):
        fs_cls = fsspec.get_filesystem_class('smb')
        self.username = username
        self.password = password
        self.host = host
        options = dict(
            host=self.host,
            username=self.username,
            password=self.password,
            port=445,
            logging_level=logging_level
            )
        self.fs = fs_cls(**options)

    @classmethod
    def from_config(cls, config: Config, host: str):
        return cls(
            username=config.username,
            password=config.password,
            host=host
        )
