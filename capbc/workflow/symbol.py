"""workflow symbol"""
from collections import OrderedDict
import json
import os
from capbc.utils import _as_list, map_aggregate, get_operator_name
from .graph_algo import post_order_dfs_visit
from .env import get_serializer


__all__ = ['Node', 'Symbol', 'group', 'save', 'dump', 'load', 'dumps', 'loads']


class Node(object):

    PLACEHOLDER_OP_TYPE = 'PLACEHOLDER'
    OPTIONAL_PLACEHOLDER_OP_TYPE = 'OPTIONAL_PLACEHOLDER'
    CONSTANT_OP_TYPE = 'CONSTANT'
    _CONSTANT_COUNT = 0

    def __init__(self, op, name, args=None, kwargs=None, attr=None):
        if attr is None:
            attr = dict()
        if '__workflow_allow_input_skip__' not in attr:
            attr['__workflow_allow_input_skip__'] = False
        if isinstance(op, str):
            assert op in [Node.PLACEHOLDER_OP_TYPE,
                          Node.OPTIONAL_PLACEHOLDER_OP_TYPE,
                          Node.CONSTANT_OP_TYPE]
        else:
            assert callable(op), f'op should be callable, but get {op}'
        self.op = op
        self._inputs = dict()
        self._name = name
        self._attr = attr
        self._args = None
        self._kwargs = None
        self._update_args_kwargs(args, kwargs)

    def __hash__(self):
        if self.is_placeholder or self.is_optional_placeholder:
            return hash(self.name) + id(self)
        else:
            return hash(self.name)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        flag = type(self) is type(other)
        if flag:
            if (self.is_placeholder or self.is_optional_placeholder) or \
                    (other.is_placeholder or other.is_optional_placeholder):
                flag &= id(self) == id(other)
            flag &= hash(self) == hash(other)
        return flag

    @property
    def is_placeholder(self):
        return isinstance(self.op, str) and self.op == Node.PLACEHOLDER_OP_TYPE

    @property
    def is_optional_placeholder(self):
        return isinstance(self.op, str) and self.op == Node.OPTIONAL_PLACEHOLDER_OP_TYPE  # noqa

    @property
    def is_constant(self):
        return isinstance(self.op, str) and self.op == Node.CONSTANT_OP_TYPE

    @property
    def num_inputs(self):
        return len(self._inputs)

    @property
    def inputs(self):
        return list(self._inputs.values())

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def name(self):
        return self._name

    @property
    def attr(self):
        return self._attr

    def wait(self, node):
        assert isinstance(node, Node)
        assert node.name != self.name, 'Cannot wait self'
        assert node not in self._inputs, f'Node {self.name} already depends on {node.name}'  # noqa
        self._inputs.setdefault(node, node)

    @classmethod
    def create_placeholder(cls, name, attr=None):
        node = cls(op=Node.PLACEHOLDER_OP_TYPE,
                   name=name, attr=attr)
        return node

    @classmethod
    def create_optional_placeholder(cls, name, default):
        node = cls(op=Node.OPTIONAL_PLACEHOLDER_OP_TYPE,
                   name=name, attr=dict(default=default))
        return node

    @classmethod
    def create_constant(cls, data, name=None):
        if name is None:
            name = 'Constant_{}{}'.format(
                get_operator_name(data), cls._CONSTANT_COUNT)
        node = cls(op=Node.CONSTANT_OP_TYPE, name=name,
                   attr=dict(data=data))
        cls._CONSTANT_COUNT += 1
        return node

    # ----- Internal Functions -----

    def _map_args(self, a, fn):
        return map_aggregate(a, lambda x: fn(x) if isinstance(x, Node) else x)

    def _update_args_kwargs(self, args=None, kwargs=None):

        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()

        new_inputs = dict()

        self._map_args(args, lambda x: new_inputs.update({x: x}))
        self._map_args(kwargs, lambda x: new_inputs.update({x: x}))
        self._args = args
        self._kwargs = kwargs

        if self.is_placeholder:
            assert len(new_inputs) == 0, 'PlaceHolder should not have inputs'

        self._inputs.update(new_inputs)


class Symbol(object):
    """
    A :py:class:`Symbol` is composed by multiple :py:class:`Node`

    Parameters
    ----------
    outputs : list/tuple of :py:class:`Node`
    """
    def __init__(self, outputs):
        self._outputs = _as_list(outputs)
        assert len(self._outputs) >= 1, 'at least one node'
        self._check()

    def _check(self):

        def check_without_duplicate_name():

            name2node = {}

            def fvisit(node):
                if node.name in name2node:
                    assert node is name2node[node.name], \
                        'Duplicate names detected: %s' % str(node.name)
                else:
                    name2node[node.name] = node

            self.post_order_dfs_visit(fvisit)

        check_without_duplicate_name()

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        return Symbol(self._outputs[idx])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return len(self._outputs)

    def get_children(self):
        outputs = []
        visited = set()
        for node in self._outputs:
            if node in visited:
                continue
            visited.add(node)
            for node in node.inputs:
                outputs.append(node)
        if not outputs:
            return None
        return Symbol(outputs)

    def get_children_name(self, recursive=False):

        name = set()

        if recursive:

            def _fvisit(node):
                assert node.name not in name
                name.add(node.name)

            self.post_order_dfs_visit(_fvisit)

        else:

            children = self.get_children()
            if children is not None:
                for child_i in children:
                    name.add(child_i.name)

        return name

    @property
    def name(self):
        names = tuple((node_i.name for node_i in self._outputs))
        return names[0] if len(names) == 1 else names

    @property
    def input_names(self):

        input_names = []

        def fvisit(node):
            if node.is_placeholder:
                input_names.append(node.name)

        self.post_order_dfs_visit(fvisit)
        return input_names

    @property
    def optional_input_names(self):

        input_names = []

        def fvisit(node):
            if node.is_optional_placeholder:
                input_names.append(node.name)

        self.post_order_dfs_visit(fvisit)
        return input_names

    @classmethod
    def create_placeholder(cls, name, attr=None):
        node = Node.create_placeholder(name, attr)
        return cls(outputs=node)

    @classmethod
    def create_optional_placeholder(cls, name, default):
        node = Node.create_optional_placeholder(name, default)
        return cls(outputs=node)

    @classmethod
    def create_constant(cls, data, name):
        node = Node.create_constant(data, name)
        return cls(outputs=node)

    def post_order_dfs_visit(self, fvisit):

        def get_input_fn(node, idx):
            key = list(node._inputs.keys())[idx]
            return node._inputs[key]

        assert callable(fvisit), 'fvisit should be callable'

        return post_order_dfs_visit(
            [output_i for output_i in self._outputs],
            visit_fn=fvisit,
            indegree_fn=lambda node: node.num_inputs,
            get_input_fn=get_input_fn,
            allow_circle=False)

    def wait(self, symbol):
        assert isinstance(symbol, Symbol)
        for out_i in self._outputs:
            for wait_i in symbol._outputs:
                out_i.wait(wait_i)
        self._check()

    def dump(self, filename):
        return dump(self, filename)

    save = dump

    @classmethod
    def load(cls, filename):
        return load(filename)

    def dumps(self):
        return dumps(self)

    @classmethod
    def loads(cls, buf):
        return loads(buf)

    def copy(self):
        return Symbol.loads(self.dumps())


def group(*symbols):
    """
    Group multiple symbols into one.
    """
    assert len(symbols) >= 1
    outputs = []
    for idx, symbol_i in enumerate(symbols):
        assert isinstance(symbol_i, Symbol), \
            'All inputs should be instance of Symbol, ' \
            'but the %d-th is an instance of %s' % (idx, str(type(symbol_i)))  # noqa
        outputs.extend(symbol_i._outputs)
    return Symbol(outputs)


def dump(symbol, filename):
    get_serializer().dump(symbol, open(filename, 'wb'))
    return True


save = dump


def load(filename):
    return get_serializer().load(open(filename, 'rb'))


def dumps(symbol):
    return get_serializer().dumps(symbol)


def loads(buf):
    return get_serializer().loads(buf)
