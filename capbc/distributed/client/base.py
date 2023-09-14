import abc


__all__ = ['ClientMixin', ]


class ClientMixin(metaclass=abc.ABCMeta):
    """
    A client can submit jobs to be execute remotely.
    """

    @property
    @abc.abstractmethod
    def num_workers(self):
        """
        Return the number of workers
        """
        pass

    @abc.abstractmethod
    def submit(self, fn, *args, **kwargs):
        """
        Submit a job.

        Parameters
        ----------
        fn : callable
            Function to be executed
        args : tuple:
            args
        kwargs : dict
            kwargs
        """
        pass
