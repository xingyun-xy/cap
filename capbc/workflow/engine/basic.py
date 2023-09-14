import os
import logging
import contextlib
from abc import ABCMeta, abstractmethod
from capbc.utils.color_message import (
    Color, ShowMethod, color_message
)
from ..exception import SkipDownStream, OpSkipFlag
from ..symbol import Node, Symbol, group


__all__ = ['BaseSymbolExecutor', 'SymbolExecutor']


logger = logging.getLogger(__name__)


CAPBC_ENABLE_LOGGING_SYMBOL_EXECUTOR = os.environ.get('CAPBC_ENABLE_LOGGING_SYMBOL_EXECUTOR', '0') == '1'  # noqa


class BaseSymbolExecutor(object):
    """
    Base symbolic graph executor.

    Parameters
    ----------
    symbol : :py:class:`Symbol`
        Symbolic graph.
    """
    def __init__(self, symbol):
        assert isinstance(symbol, Symbol), \
            f'Expected {Symbol}, but get {type(symbol)}'
        self.symbol = symbol

    @abstractmethod
    def __call__(self, inputs):
        pass


class SymbolExecutor(BaseSymbolExecutor):
    """
    Simple executor to execute an symbolic graph, using
    dfs visit.
    """
    def __init__(self, symbol, verbose=None):
        super().__init__(symbol)
        self._head2child_map = dict()
        for symbol_i in self.symbol:
            self._head2child_map[symbol_i.name] = \
                set(symbol_i.get_children_name(recursive=True))
        if verbose is None:
            verbose = CAPBC_ENABLE_LOGGING_SYMBOL_EXECUTOR
        self.verbose = verbose

    def __call__(self, inputs, caches=None):
        """
        Executing graph using dfs visit. When a :py:class:`capbc.workflow.exception.SkipDownStream` exception
        occurs, will output a `OpSkipFlag` object.

        Parameters
        ----------
        inputs : dict
            Inputs, should contains all placeholder values.
        """  # noqa
        assert isinstance(inputs, dict), \
            f'Expected dict, but get {type(inputs)}'

        if caches is None:
            caches = dict()
        assert isinstance(caches, dict), \
            f'Expected dict, but get {type(caches)}'

        node_outputs = dict()

        def _has_skip(node):
            for input_i in node.inputs:
                if node_outputs[input_i.name] is OpSkipFlag:
                    return True
            return False

        def fvisit(node):
                
            if node.name in node_outputs:
                return
            if node.is_optional_placeholder:
                if node.name not in inputs:
                    node_outputs[node.name] = node.attr['default']
                else:
                    node_outputs[node.name] = inputs[node.name]
                if self.verbose:
                    logger.info(f'{node.name} is OptionalVariable, value={node_outputs[node.name]}')  # noqa
            elif node.is_placeholder:
                assert node.name in inputs, \
                    f'Missing input for placeholder node {node.name}'
                node_outputs[node.name] = inputs[node.name]
                if self.verbose:
                    logger.info(f'{node.name} is Variable, value={node_outputs[node.name]}')  # noqa
            elif node.is_constant:
                node_outputs[node.name] = node.attr['data']
                if self.verbose:
                    logger.info(f'{node.name} is Constant, value={node_outputs[node.name]}')  # noqa
            else:
                if _has_skip(node) and not node.attr['__workflow_allow_input_skip__']:  # noqa
                    node_outputs[node.name] = OpSkipFlag
                    if self.verbose:
                        logger.info(f'Some input of node={node.name} is skip, skip this node')  # noqa
                    return
                try:
                    args = node._map_args(node.args, lambda x: node_outputs[x.name])  # noqa
                    kwargs = node._map_args(node.kwargs, lambda x: node_outputs[x.name])  # noqa
                    if self.verbose:
                        logger.info(f'Begin executing {node.name}, op={node.op}, args={args}, kwargs={kwargs}')  # noqa
                    ret = node.op(*args, **kwargs)
                    assert ret is not OpSkipFlag, "Directly return `OpSkipFlag` is not allowed, raise exception SkipDownStream instead."  # noqa
                    node_outputs[node.name] = ret
                except SkipDownStream as e:
                    node_outputs[node.name] = OpSkipFlag
                    msg = f'Skip down stream of node={node.name}, op={node.op}, args={args}, kwargs={kwargs}'  # noqa
                    if e.args:
                        msg += f', reason={e.args}'
                    msg = color_message(msg, show_method=ShowMethod.highlight,
                                        background_color=Color.red)
                    logger.info(f'\n\n{msg}\n\n')
                except Exception as e:
                    logger.error(f'Executing {node.name}, op={node.op}, args={args}, kwargs={kwargs} Failed! Exception: {e}')  # noqa
                    raise e
                if self.verbose:
                    logger.info(f'End executing {node.name}, op={node.op}, '
                        f'args={args}, kwargs={kwargs}, outputs={node_outputs[node.name]}\n')  # noqa

        def inner(node):
            fvisit(node)
            caches[node.name] = node_outputs[node.name]

        self.symbol.post_order_dfs_visit(inner)

        # get head output
        outputs = []
        for node in self.symbol._outputs:
            if node.name not in node_outputs:
                outputs.append(OpSkipFlag)
            else:
                outputs.append(node_outputs[node.name])
        return outputs[0] if len(outputs) == 1 else outputs
