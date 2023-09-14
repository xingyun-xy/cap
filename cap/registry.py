# Copyright (c) Changan Auto. All rights reserved.

import inspect
import pickle
from collections import defaultdict
from typing import Any, Dict, Union
from mmcv.utils import Registry as _Registry

import torch
import torchvision

__all__ = [
    "Registry",
    "OBJECT_REGISTRY",
    "build_from_cfg",
    "build_from_registry",
    "RegistryContext",
]

class Registry(_Registry):
    def __init__(self, name: str):
        self._name = name
        self._name_obj_map = {}
        self._lower_str = False

    def pop(self, name, default=None):
        return self._name_obj_map.pop(name, default)

    def delete(self, name):
        self.pop(name, None)

    def _get_name(self, name):
        if isinstance(name, str) and self._lower_str:
            name = name.lower()
        return name

    def __contains__(self, name):
        name = self._get_name(name)
        return name in self._name_obj_map

    def keys(self):
        return self._name_obj_map.keys()

    def _do_register(self, name, obj):
        name = self._get_name(name)
        assert (name not in self._name_obj_map), \
            "An object named '{}' was already registered in '{}' registry!".format(  # noqa
            name, self._name)
        self._name_obj_map[name] = obj

    def register(self, obj=None, *, name=None):
        """
        Register the given object under the the name `obj.__name__`
        or given name.
        """
        if obj is None and name is None:
            raise ValueError('Should provide at least one of obj and name')
        if obj is not None and name is not None:
            self._do_register(name, obj)
        elif obj is not None and name is None:  # used as decorator
            name = obj.__name__
            self._do_register(name, obj)
            return obj
        else:
            return self.alias(name)

    def register_module(self, *args, **kwargs):  # type: ignore
        return self.register(*args, **kwargs)

    def alias(self, name):
        """Get registrator function that allow aliases.

        Parameters
        ----------
        name: str
            The register name

        Returns
        -------
        a registrator function
        """

        def reg(obj):

            self._do_register(name, obj)
            return obj

        return reg

    def get(self, name):
        origin_name = name
        name = self._get_name(name)
        ret = self._name_obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    origin_name, self._name
                )
            )
        return ret


OBJECT_REGISTRY = Registry('CAP_OBJECT_REGISTRY')


def build_from_cfg(registry: Registry, cfg: dict) -> Any:
    if not isinstance(registry, Registry):
        raise TypeError("Expected Registry, but get {}".format(type(registry)))
    if not isinstance(cfg, dict):
        raise TypeError("Expected dict, but get {}".format(type(cfg)))
    if "type" not in cfg:
        raise KeyError("Required has key `type`, but not")
    cfg = cfg.copy()

    cfg.pop("__graph_model_name", None)
    cfg.pop("__build_recursive", None)

    obj_type = cfg.pop("type")
    if obj_type in registry:
        obj_cls = registry.get(obj_type)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        _raise_invalid_type_error(obj_type, registry)
    try:
        instance = obj_cls(**cfg)
    except TypeError as te:
        raise TypeError("%s: %s" % (obj_cls, te))
    return instance


def _build_optimizer(cfg: dict) -> Any:
    def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
        if "params" in cfg:
            loc_name = {}  # type: ignore
            for k, v in cfg["params"].items():
                loc_name[k] = {"params": []}
                loc_name[k].update(v)

            loc_name["others"] = {
                "params": [],
                "weight_decay": (
                    cfg["weight_decay"] if "weight_decay" in cfg else 0
                ),
            }
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    pass
                flag = False
                for k, _v in cfg["params"].items():
                    if k in name:
                        loc_name[k]["params"].append(p)
                        flag = True
                        break
                if not flag:
                    loc_name["others"]["params"].append(p)

            res = []
            for _k, v in loc_name.items():
                res.append(v)
            cfg["params"] = res
        else:
            cfg["params"] = filter(
                lambda p: p.requires_grad, model.parameters()
            )
        return build_from_cfg(OBJECT_REGISTRY, cfg)

    if "model" in cfg:
        model = cfg.pop("model")
        assert isinstance(model, torch.nn.Module)
        return build_optimizer(model)
    else:
        return build_optimizer


def _as_list(obj):
    """A utility function that converts the argument to a list if it is not
    already.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list or tuple, return it. Otherwise, return `[obj]` as a
    single-element list.

    """
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


def _raise_invalid_type_error(obj_type, registry=None):  # type: ignore
    if registry is None:
        registry = [
            OBJECT_REGISTRY,
        ]
    registry_names = [registry_i._name for registry_i in _as_list(registry)]
    raise TypeError(
        "{} has not registered in any of registry {} and is not a class, which is not allowed".format(  # noqa
            obj_type, registry_names
        )
    )

def _modify_pytorch_dataloader_config(cfg: dict) -> dict:
    if "sampler" in cfg:
        if (
            isinstance(cfg["sampler"], dict)
            and "dataset" not in cfg["sampler"]
        ):  # noqa
            cfg["sampler"]["dataset"] = cfg["dataset"]
        cfg["shuffle"] = False
    return cfg

class RegistryContext:
    """Store the mapping between object id and object instance."""

    _current: Union[None, Dict] = None

    def __init__(self) -> None:
        self._old = None

    def __enter__(self):  # type: ignore
        assert RegistryContext._current is None
        self._old = RegistryContext._current
        RegistryContext._current = dict()  # noqa
        return self

    def __exit__(self, value, ptype, trace):  # type: ignore
        RegistryContext._current = self._old

    @classmethod
    def get_current(cls) -> Union[None, Dict]:
        return cls._current


def _build_dataset(cfg: dict) -> Any:
    if "transforms" in cfg and cfg["transforms"] is not None:
        if isinstance(cfg["transforms"], (list, tuple)):
            cfg["transforms"] = torchvision.transforms.Compose(
                cfg["transforms"]
            )  # noqa
    obj = build_from_cfg(OBJECT_REGISTRY, cfg)
    obj = pickle.loads(pickle.dumps(obj))
    return obj


def build_from_registry(x: Any) -> Any:
    """
    Build object from registry.

    This function will recursively visit all elements, if an object is dict
    and has the key `type`, which is considered as an object that should be
    build.
    """

    def _impl(x):  # type: ignore
        id2object = RegistryContext.get_current()
        if isinstance(x, (list, tuple)):
            x = type(x)((_impl(x_i) for x_i in x))
            return x
        elif isinstance(x, dict):
            object_id = id(x)
            has_type = "type" in x
            object_type = x.get("type", None)
            if has_type and object_id in id2object:
                return id2object[object_id]
            if has_type and object_type is torch.utils.data.DataLoader:
                x = _modify_pytorch_dataloader_config(x)

            if x.get("__build_recursive", True):
                # fmt: off
                build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
                # fmt: on
            else:
                build_x = x

            if type(x) is defaultdict:
                x = defaultdict(x.default_factory, build_x)
            else:
                x = type(x)(build_x)

            if has_type:
                if object_type in OBJECT_REGISTRY:
                    object_type = OBJECT_REGISTRY.get(object_type)
                    isclass = inspect.isclass(object_type)
                elif inspect.isclass(object_type):
                    isclass = True
                else:
                    _raise_invalid_type_error(object_type)
                if isclass and issubclass(
                    object_type, torch.utils.data.Dataset
                ):  # noqa
                    obj = _build_dataset(x)
                elif isclass and issubclass(
                    object_type, torch.optim.Optimizer
                ):  # noqa
                    obj = _build_optimizer(x)
                else:
                    obj = build_from_cfg(OBJECT_REGISTRY, x)
                id2object[object_id] = obj
                return obj
            else:
                return x
        else:
            return x

    current = RegistryContext.get_current()
    if current is None:
        with RegistryContext():
            return _impl(x)
    else:
        return _impl(x)

def build_from_registry(x: Any, rebuild=False) -> Any:
    """
    Build object from registry.

    This function will recursively visit all elements, if an object is dict
    and has the key `type`, which is considered as an object that should be
    build.
    """

    def _impl(x):  # type: ignore
        id2object = RegistryContext.get_current()
        if isinstance(x, (list, tuple)):
            x = type(x)((_impl(x_i) for x_i in x))
            return x
        elif isinstance(x, dict):
            object_id = id(x)
            has_type = "type" in x
            object_type = x.get("type", None)
            if has_type and object_id in id2object and rebuild == False:
                return id2object[object_id]
            if has_type and object_type is torch.utils.data.DataLoader:
                x = _modify_pytorch_dataloader_config(x)

            if x.get("__build_recursive", True):
                # TODO(zhangwenjie): add unittest
                # fmt: off
                build_x = dict(((key, _impl(value)) for key, value in x.items()))  # noqa
                # fmt: on
            else:
                build_x = x

            if type(x) is defaultdict:
                x = defaultdict(x.default_factory, build_x)
            else:
                x = type(x)(build_x)

            if has_type:
                if object_type in OBJECT_REGISTRY:
                    object_type = OBJECT_REGISTRY.get(object_type)
                    isclass = inspect.isclass(object_type)
                elif inspect.isclass(object_type):
                    isclass = True
                else:
                    _raise_invalid_type_error(object_type)
                if isclass and issubclass(
                    object_type, torch.utils.data.Dataset
                ):  # noqa
                    obj = _build_dataset(x)
                elif isclass and issubclass(
                    object_type, torch.optim.Optimizer
                ):  # noqa
                    obj = _build_optimizer(x)
                else:
                    obj = build_from_cfg(OBJECT_REGISTRY, x)
                id2object[object_id] = obj
                return obj
            else:
                return x
        else:
            return x

    current = RegistryContext.get_current()
    if current is None:
        with RegistryContext():
            return _impl(x)
    else:
        return _impl(x)