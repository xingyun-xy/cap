from capbc.registry import Registry
from capbc.data.backend import Backend


__all__ = [
    "Backend",
    "BACKEND_OP_REGISTRY",
    "get_backend_op_register_name"
]



BACKEND_OP_REGISTRY = Registry("BACKEND_OP")


def get_backend_op_register_name(backend, cls):
    return (Backend(backend), cls)
