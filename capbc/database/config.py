import os

import yaml

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/hdflow-database.yaml")


def load_config(path=DEFAULT_CONFIG_PATH):
    assert os.path.exists(path), f"Missing {path}"
    return yaml.safe_load(open(path, "r"))
