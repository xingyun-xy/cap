from functools import partial, wraps
from abc import abstractmethod
from capbc.utils import map_aggregate
from capbc.workflow.symbol import (
    Node,
    Symbol
)
from capbc.data.datapipe.utils import (
    DataPipeOp,
    NameStorage,
)


__all__ = ["DataPipe", "zip", "from_sequence"]


name_storage = NameStorage()


def _create_datapipe(name, *args, **kwargs):
    """Create :py:class:`DataPipe`.

    .. note::

        For internal use only.
    """

    def _map_func(x):
        if isinstance(x, DataPipe):
            return x.graph._outputs[0]
        else:
            return x

    args = map_aggregate(args, _map_func)
    kwargs = map_aggregate(kwargs, _map_func)

    name_storage.add(name)

    symbol = Symbol(
        outputs=[
            Node(
                op=DataPipeOp(name),
                name=name_storage.get(name),
                args=args,
                kwargs=kwargs
            )
        ]
    )
    return DataPipe(symbol)


def datapipe_op(op=None, *, method_type="naive", eager=False):

    if op is None:
        return partial(datapipe_op, method_type=method_type, eager=eager)

    @wraps(op)
    def _inner(*args, **kwargs):

        if method_type == "class":
            assert len(args) >= 1
            args = args[1:]

        ret = _create_datapipe(op.__name__, *args, **kwargs)
        if eager:
            return ret.compute()
        else:
            return ret

    return _inner


class DataPipe(object):
    """Distributed data processing pipeline."""

    def __init__(self, graph: Symbol):
        """

        .. note::

            Do not initialized directly, using `from_sequence` instead.

        Parameters
        ----------
        graph : Symbol
            The computational graph.
        """
        assert isinstance(graph, Symbol)
        assert len(graph) == 1, "At most one outputs"
        self.graph = graph

    @property
    def executor(self):

        from capbc.data.executor import DataPipeExecutor

        return DataPipeExecutor.get_current()

    # computation

    def compute(self):
        """Compute the datapipe."""
        return self.executor.compute(self)

    def __iter__(self):
        yield from self.compute()

    def iter_batches(self, batch_size: int):
        """Return an iterator that each time return `batch_size` samples.

        Parameters
        ----------
        batch_size : int
            Batch size
        """
        return self.executor.iter_batches(self, batch_size)

    # operators

    @datapipe_op
    def cache(self) -> "DataPipe":
        """Cache results in memory.

        Examples
        --------
        >>> datapipe = datapipe.cache()
        """

    @datapipe_op
    def map(self, func) -> "DataPipe":
        """Apply a function elementwise.

        Examples
        --------
        >>> datapipe = DataPipe.from_sequence([1, 2, 3, 4], npartitions=2)
        >>> datapipe = datapipe.map(lambda x: x + 2)
        >>> executor = DataPipeExecutor("dask", 2)
        >>> executor.compute(datapipe)
        (3, 4, 5, 6)
        """

    @datapipe_op
    def map_partitions(self, func) -> "DataPipe":
        """Apply a function to every partition.

        Examples
        --------
        >>> datapipe = DataPipe.from_sequence([0, 1, 2, 3, 4, 5], npartitions=2)
        >>> datapipe = datapipe.map_partitions(sum)
        >>> executor = DataPipeExecutor("dask", 2)
        >>> executor.compute(datapipe)
        (3, 12)
        """  # noqa

    @datapipe_op
    def map_batches(self, func, batch_size) -> "DataPipe":
        """Apply a function to `batch_size` samples.

        Examples
        --------
        >>> datapipe = DataPipe.from_sequence([0, 1, 2, 3, 4, 5], npartitions=2)
        >>> datapipe = datapipe.map_batches(sum, batch_size=2)
        >>> executor = DataPipeExecutor("dask", 2)
        >>> executor.compute(datapipe)
        (1, 2, 7, 5)
        """  # noqa

    @datapipe_op
    def filter(self, func) -> "DataPipe":
        """Filter out elements.

        Examples
        --------
        >>> datapipe = DataPipe.from_sequence([0, 1, 2, 3, 4])
        >>> datapipe = datapipe.filter(lambda x: x % 2 == 0)
        >>> executor = DataPipeExecutor("dask", 4)
        >>> print(executor.compute(datapipe))
        (0, 2, 4)
        """

    @datapipe_op
    def zip(self, *datapipes) -> "DataPipe":
        """Partition-wise zip.

        All datapipes must have the same number of partitions.

        Examples
        --------
        >>> datapipe1 = DataPipe.from_sequence([0, 1, 2, 3], npartitions=2)
        >>> datapipe2 = DataPipe.from_sequence([10, 11, 12, 13], npartitions=2)
        >>> datapipe3 = DataPipe.from_sequence([100, 101, 102, 103], npartitions=2)
        >>> datapipe = datapipe3.zip(datapip1, datapipe2)
        >>> executor = DataPipeExecutor("dask", 2)
        >>> executor.compute(datapipe)
        ((0, 10, 100), (1, 11, 101), (2, 12, 102), (3, 13, 103))
        """  # noqa

    @datapipe_op
    def repartition(self, npartitions) -> "DataPipe":
        """Repartition `DataPipe`.

        Examples
        --------
        >>> datapipe = datapipe.repartition(4)  # new datapipe has 4 partitions
        """

    @classmethod
    @datapipe_op(method_type="class")
    def from_sequence(cls, data, npartitions=1) -> "DataPipe":
        """Create :py:class:`DataPipe` by iterable objects.

        Examples
        --------
        >>> datapipe = DataPipe.from_sequence([0, 1, 2, 3], npartitions=2)
        >>> executor = DataPipeExecutor("dask", 2)
        >>> executor.compute(datapipe)
        (0, 1, 2, 3)
        """


def zip(*datapipes) -> DataPipe:
    """See the `zip` method of :py:class:`DataPipe`."""
    if not datapipes:
        raise ValueError("At least one datapipe")
    elif len(datapipes) == 1:
        return datapipes[0]
    else:
        return datapipes[0].zip(*datapipes[1:])


def from_sequence(data, npartitions=1) -> DataPipe:
    """See the `from_sequence` method of :py:class:`DataPipe`."""
    return DataPipe.from_sequence(data, npartitions)
