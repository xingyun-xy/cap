from .base import ClientMixin


class _MainProcessFuture:
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def result(self, timeout=None):
        return self.fn(*self.args, **self.kwargs)

    def done(self):
        return True


class MainProcessClient(ClientMixin):

    @property
    def num_workers(self):
        return 1

    def submit(self, fn, *args, **kwargs):
        return _MainProcessFuture(fn, args, kwargs)
