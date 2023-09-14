import signal
import functools
import logging
from .kthread import KThread


__all__ = ['timeout_call', 'timeout']


logger = logging.getLogger(__name__)


def timeout_call(seconds, func, args=tuple(), kwargs=None,
                 message=None):
    """
    Call a function and raise a TimeoutError when executing time
    exceed timeout duration.

    Parameters
    ----------
    seconds : float
        In seconds
    func : callable
    args : tuple, optional
        Function args, by default tuple()
    kwargs : dict, optional
        Function kwargs, by default None
    message : str, optional
        TimeoutError message, by default None
    """

    if message is None:
        message = f'Calling {func} take time exceed {seconds}s'

    if kwargs is None:
        kwargs = dict()

    result = []
    exception = None

    def _new_func(oldfunc, result, oldfunc_args, oldfunc_kwargs):
        try:
            result.append(oldfunc(*oldfunc_args, **oldfunc_kwargs))
        except BaseException as e:
            nonlocal exception
            exception = e

    # create new args for _new_func, because we want to get the func return val to result list  # noqa
    new_kwargs = {
        'oldfunc': func,
        'result': result,
        'oldfunc_args': args,
        'oldfunc_kwargs': kwargs
    }

    thd = KThread(target=_new_func, args=(), kwargs=new_kwargs)

    thd.start()

    thd.join(seconds)

    alive = thd.isAlive()

    thd.kill()  # kill the child thread

    if alive:
        raise TimeoutError(message)
    elif exception is not None:
        raise exception
    else:
        return result[0]


def timeout(seconds, message=None):
    """
    A decorator for timeout_call
    """

    def timeout_decorator(func):

        @functools.wraps(func)
        def _inner(*args, **kwargs):
            return timeout_call(
                seconds, func=func, args=args,
                kwargs=kwargs, message=message)

        return _inner

    return timeout_decorator
