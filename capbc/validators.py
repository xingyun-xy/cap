from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Optional


__all__ = ['BaseValidator', 'IsInstance', 'IsListOfInstance',
           'AnyOf', 'Length', 'Range']


class BaseValidator(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, data) -> Tuple[bool, str]:
        pass


class IsInstance(BaseValidator):

    def __init__(self, *valid_types: Union[type, Tuple[type]]):
        if isinstance(valid_types, list):
            valid_types = tuple(valid_types)
        elif not isinstance(valid_types, tuple):
            valid_types = (valid_types, )
        self._valid_types = valid_types

    def __call__(self, data):
        if not isinstance(data, self._valid_types):
            success = False
            msg = f'Must be a instance of: {self._valid_types}, but get {type(data)}'  # noqa
        else:
            success = True
            msg = 'ok'
        return success, msg


class IsListOfInstance(BaseValidator):

    def __init__(self, *valid_types: Union[type, Tuple[type]]):
        self._valid_types = valid_types
        self._list_validator = IsInstance(tuple, list)
        self._is_instance_validator = IsInstance(*valid_types)

    def __call__(self, data):
        success, msg = self._list_validator(data)
        if success:
            for data_i in data:
                success, _ = self._is_instance_validator(data_i)
                if not success:
                    msg = f'Must be a list of instance of: {self._valid_types}, but get {type(data_i)}'  # noqa
                    break
            else:
                msg = 'ok'
        return success, msg


class AnyOf(BaseValidator):

    def __init__(self, values: list):
        self._values = values

    def __call__(self, data):
        if data not in self._values:
            success = False
            msg = f'Must be one of: {self._values}, but get {data}'
        else:
            success = True
            msg = 'ok'
        return success, msg


class Length(BaseValidator):

    def __init__(self, min: Optional[int] = None, max: Optional[int] = None):
        self._min = min
        self._max = max

    def __call__(self, data):
        if self._min is not None and len(data) < self._min:
            success = False
            msg = f'Length must not less than {self._min}, but get {data}'
        elif self._max is not None and len(data) > self._max:
            success = False
            msg = f'Length must not greater than {self._max}, but get {data}'
        else:
            success = True
            msg = 'ok'
        return success, msg


class Range(BaseValidator):
    def __init__(self, lt=None, le=None, gt=None, ge=None):
        self._lt = lt
        self._le = le
        self._gt = gt
        self._ge = ge

    def __call__(self, data):
        if self._lt is not None and not data < self._lt:
            success = False
            msg = f'Must less than {self._lt}, but get {data}'
        elif self._le is not None and not data <= self._le:
            success = False
            msg = f'Must less than or equal to {self._le}, but get {data}'
        elif self._gt is not None and not data > self._gt:
            success = False
            msg = f'Must greater than or equal to {self._gt}, but get {data}'
        elif self._ge is not None and not data >= self._ge:
            success = False
            msg = f'Must greater than or equal to {self._ge}, but get {data}'
        else:
            success = True
            msg = 'ok'
        return success, msg
