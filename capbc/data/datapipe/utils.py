import logging

from capbc.data.backend import Backend
from capbc.data.op.base import (
    BACKEND_OP_REGISTRY,
    get_backend_op_register_name
)


logger = logging.getLogger(__name__)


class NameStorage:
    def __init__(self):
        self._name2count = dict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_name2count"] = dict()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()

    def add(self, name: str):
        if name not in self._name2count:
            self._name2count[name] = 0
        self._name2count[name] += 1

    def get(self, name: str) -> str:
        assert name in self._name2count
        return "{}_{}".format(name, self._name2count[name])


class DataPipeOp:

    backend: Backend = None

    def __init__(self, op):
        self.op = op

    @classmethod
    def set_backend(cls, backend):
        assert isinstance(backend, Backend)
        cls.backend = backend

    @classmethod
    def reset_backend(cls):
        cls.backend = None

    def __call__(self, *args, **kwargs):

        assert DataPipeOp.backend is not None, "Please set backend first!"

        backend_op_register_name = get_backend_op_register_name(
            DataPipeOp.backend, self.op
        )
        if backend_op_register_name not in BACKEND_OP_REGISTRY:
            raise NotImplementedError(
                "Backend {} does not support operator `{}` yet.".format(
                    DataPipeOp.backend,
                    self.op.__name__
                )
            )
        op = BACKEND_OP_REGISTRY.get(backend_op_register_name)

        try:
            ret = op(*args, **kwargs)
        except Exception as e:
            raise type(e)(f"Except when executing op `{self.op}`, error messge = {e}")  # noqa

        return ret
