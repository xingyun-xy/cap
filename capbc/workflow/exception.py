

__all__ = ['SkipDownStream', 'OpSkipFlag']


class SkipDownStream(Exception):
    """
    For operators that want to skip down stream task, raise this exception.

    .. note::

        This only works when :py:class:`capbc.workflow.trace.GraphTracer` is active.
    """  # noqa
    pass


class OpSkipFlag:
    """
    The output value for skipped operators
    """
    pass
