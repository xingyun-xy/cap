from __future__ import absolute_import, print_function
import os
import sys
from importlib import import_module
import json
import yaml


__all__ = ['load_pyfile', 'load_json', 'load_yaml']


# TODO: using filestream to read


def _check_path_exists(path, local_only=False):
    assert os.path.exists(path), '%s does not exists' % path


def load_pyfile(filename, allow_unsafe=False, pop_module=True):
    """
    Load python file.
    """
    _check_path_exists(filename, local_only=True)
    module_name = os.path.basename(filename)[:-3]
    config_dir = os.path.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    if pop_module:
        sys.modules.pop(module_name)
    return cfg


def load_json(filename, allow_unsafe=False):
    """
    Load json file
    """
    _check_path_exists(filename)
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg


def load_yaml(filename, allow_unsafe=False):
    """
    Load yaml file
    """
    _check_path_exists(filename)
    with open(filename, 'r') as f:
        if allow_unsafe:
            cfg = yaml.unsafe_load(f)
        else:
            cfg = yaml.safe_load(f)
    return cfg


ext_to_load_fn_map = {
    '.py': load_pyfile,
    '.json': load_json,
    '.yaml': load_yaml,
    '.yml': load_yaml
}
