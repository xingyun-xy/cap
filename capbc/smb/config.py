import os
from typing import Optional
from dataclasses import dataclass, fields
from dataclasses_json import DataClassJsonMixin


@dataclass
class Config(DataClassJsonMixin):
    """
    Config of SMBClient.

    Parameters
    ----------
    username : Optional[str]
        Username of Atlassian account.
    password : Optional[str]
        Password of Atlassian account.
    """
    username: Optional[str] = None
    password: Optional[str] = None

    def update(self, **kwargs):
        _field_names = set([field.name for field in fields(self)])
        for key in kwargs:
            if key in _field_names:
                if kwargs[key] is not None:
                    setattr(self, key, kwargs[key])
            else:
                raise ValueError('Unexpected param: %s' % key)


def _get_default_config_path():
    default_config_path = os.path.expanduser(
        os.path.join('~', '.config', 'smb', 'config.yaml'))
    return default_config_path


DEFAULT_CONFIG_PATH = _get_default_config_path()
