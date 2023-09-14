from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Callable, ClassVar, Union

import torch


@dataclass
class BaseData(object):
    """Base class for all single-sample data structures."""

    @classmethod
    def combine(cls, *insts):
        assert len(cls.__bases__) == len(insts)
        all_data = {}
        for base, inst in zip(cls.__bases__, insts):
            assert isinstance(inst, base) and is_dataclass(base)
            for f in fields(inst):
                assert f.name not in all_data
                all_data[f.name] = getattr(inst, f.name)

        return cls(**all_data)

    @property
    def _major(self):
        for field in fields(type(self)):
            ret = getattr(self, field.name)
            if ret is not None and isinstance(
                ret, (BaseData, torch.Tensor, list)
            ):
                return ret
        raise ValueError("SHOULD NOT BE ALL None")

    @property
    def device(self):
        return self._major.device

    def _to(self, device, **kwargs):
        cur_cls = type(self)
        data_fields = fields(cur_cls)
        for f in data_fields:
            val = getattr(self, f.name)
            if val is not None:
                try:
                    if isinstance(val, list):
                        val = [v.to(device) for v in val]
                    else:
                        val = val.to(device)
                except AttributeError:
                    val = val
                kwargs[f.name] = val

        return cur_cls(**kwargs)

    def to(self, device):
        return self._to(device)

    def rescale(self, *args, **kwargs):
        cur_cls = type(self)
        data_fields = fields(cur_cls)
        _kwargs = {}
        for f in data_fields:
            val = getattr(self, f.name)
            if isinstance(val, BaseData):
                val = val.rescale(*args, **kwargs)
                _kwargs[f.name] = val

        return cur_cls(**_kwargs)

    def to_sda_eval(self):
        raise NotImplementedError


@dataclass
class BaseDataList(BaseData):
    """Base class for all multiple-sample data structures."""

    singleton_cls: ClassVar[BaseData]

    def _getitem(self, idx: Union[int, slice, torch.Tensor], **kwargs):

        cur_cls = type(self)
        cur_fields = fields(cur_cls)

        singleton_cls = self.singleton_cls
        singleton_fields = fields(singleton_cls)

        args = [
            getattr(self, f.name)[idx]
            for f in cur_fields
            if f.default is MISSING
        ]

        for cf, sf in zip(cur_fields, singleton_fields):
            if cf.default is MISSING:
                continue

            val = getattr(self, cf.name)
            if val is not None:
                name = cf.name if not isinstance(idx, int) else sf.name
                if isinstance(val, list):
                    # val list condition
                    if isinstance(idx, int):
                        kwargs[name] = val[idx]
                        continue

                    if torch.is_tensor(idx):
                        # scalar tensor
                        if idx.ndim == 0:
                            kwargs[name] = val[idx.item()]
                            break
                        assert idx.ndim == 1

                        # bool tensor
                        if idx.dtype == torch.bool:
                            idx = torch.where(idx)[0]

                        idx_list = idx.cpu().numpy().tolist()

                        kwargs[name] = [val[i] for i in idx_list]
                else:
                    kwargs[name] = val[idx]

        ret_cls = cur_cls if not isinstance(idx, int) else singleton_cls

        return ret_cls(*args, **kwargs)

    def __getitem__(self, idx: Union[int, slice, torch.Tensor]):
        return self._getitem(idx)

    def __len__(self):
        return len(self._major)

    @property
    def device(self):
        if isinstance(self._major, list):
            return self._major[0]
        return self._major.device

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < len(self):
            ret = self[self._iter_idx]
            self._iter_idx += 1
            return ret
        else:
            raise StopIteration

    def filter_by_lambda(self, func: Callable):
        indices = func(self)
        return self[indices]

    def to_sda_eval(self):
        iter(self)
        return [data.to_sda_eval() for data in self]

    def inv_pad(self, *padding):
        cur_cls = type(self)
        data_fields = fields(cur_cls)
        _kwargs = {}
        for f in data_fields:
            val = getattr(self, f.name)
            if isinstance(val, BaseDataList):
                val = val.inv_pad(*padding)
                _kwargs[f.name] = val

        return cur_cls(**_kwargs)
