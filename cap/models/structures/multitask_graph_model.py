# Copyright (c) Changan Auto. All rights reserved.
import numpy as np
import logging
import threading
from collections import ChainMap, OrderedDict, defaultdict, namedtuple
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from capbc.workflow.engine import SymbolExecutor
from capbc.workflow.proxy import (
    OptionalVariable,
    Variable,
    WorkflowVariable,
    get_traced_graph,
)
from capbc.workflow.symbol import Node, Symbol, group
from capbc.workflow.trace import GraphTracer

from cap.registry import OBJECT_REGISTRY, build_from_registry
from cap.utils.apply_func import (
    _as_list,
    flatten,
    is_list_of_type,
    regroup,
    to_flat_ordered_dict,
)

__all__ = ["MultitaskGraphModel"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class MultitaskGraphModel(nn.Module):

    # GraphTracer is not threading safe #
    TRACER_LOCK = threading.Lock()
    GraphTracer.register_basic_types(torch.Tensor)

    def __init__(
        self,
        inputs: Dict[str, Any],
        task_inputs: Dict[str, Any],
        task_modules: Dict[str, nn.Module],
        opt_inputs: Optional[Dict[str, Any]] = None,
        funnel_modules: Optional[Dict[str, nn.Module]] = None,
        flatten_outputs: bool = True,
        lazy_forward: Optional[bool] = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self._cached_execs = {}

        self.flatten_outputs = flatten_outputs
        self._output_names2flat_conditions = {}

        task_names = list(task_inputs.keys())

        # get local rank
        rank = torch.cuda.current_device()
        device = torch.device(f"cuda:{rank}")

        if funnel_modules is None:
            funnel_modules = {}

        # nodes and modules
        nodes = self._build_nodes(ChainMap(task_modules, funnel_modules))
        task_modules = {
            k: build_from_registry(v).to(device)
            for k, v in task_modules.items()
        }

        funnel_modules = {
            k: build_from_registry(v).to(device)
            for k, v in funnel_modules.items()
        }

        opt_inputs = {} if opt_inputs is None else opt_inputs
        inputs_var, task_inputs_var = self._build_variables(
            inputs, opt_inputs, task_inputs, lazy_forward, device=device
        )

        self._name2fmts, self._cached_graphs = OrderedDict(), OrderedDict()

        # 1. build: synchronously execute GraphTracer, since it is not
        # threading safe.
        with self.TRACER_LOCK, GraphTracer(imperative=not lazy_forward):
            tmp_outputs = OrderedDict()
            # 遍历每个task，找到task module和task input
            for task in task_names:
                _task_module = task_modules[task]
                t_inputs = task_inputs_var[task][task]
                input_dict = {k: t_inputs[k] for k in t_inputs.keys()}
                input_dict.update(inputs_var)   # 将taskinput和input合并到一起作为输入
                tmp_outputs[task] = _task_module(input_dict)

            for (input_names, out_name), module in funnel_modules.items():
                funnel_inputs = [tmp_outputs.pop(name) for name in input_names]
                tmp_outputs[out_name] = module(*funnel_inputs)

            st = 0
            for out_name, _outputs in tmp_outputs.items():
                _outputs, _formats = flatten(_outputs)
                self._name2fmts[out_name] = (
                    (st, st + len(_outputs)),
                    _formats,
                )
                st += len(_outputs)

                for obj in _outputs:
                    assert isinstance(obj, WorkflowVariable), (
                        "graph outputs should be one or a sequence of "
                        "`WorkflowVariable` objects, but get %s" % type(obj)
                    )

                self._cached_graphs[out_name] = get_traced_graph(_outputs)

            self._output_names = list(tmp_outputs.keys())

        del tmp_outputs
        torch.cuda.empty_cache()

        symbols = list(self._cached_graphs.values())

        self._merge_nodes(symbols)
        self._graph = group(*symbols)

        if transform is not None:
            self._transform_graph(transform)

        node2name = {v: k for k, v in nodes.items()}

        self._node_name_2_mod_name = {}

        # 2. register modules by traversing the traced graph
        self._register_nodes(node2name)

        # 3. check if all trainable parameters and buffers are covered
        self._check_params_and_buffers(ChainMap(task_modules, funnel_modules))
        self.cpu()

        # set default flat_conditions
        for name in self._output_names:
            if name not in self._output_names2flat_conditions:
                self._output_names2flat_conditions[
                    name
                ] = lambda _, v: not is_list_of_type(v, torch.Tensor)

    @staticmethod
    def _build_nodes(module_dict):
        _nodes = {}

        def _is_node_cfg(item):
            return (
                isinstance(item, dict)
                and "type" in item
                and "__graph_model_name" in item
            )

        def _check_cfg(item, allow_node=True):
            _allow_node = True
            if _is_node_cfg(item):
                if not allow_node:
                    raise ValueError(f"{item} node in node")
                name = item["__graph_model_name"]
                # assert no duplication
                if name in _nodes:
                    assert id(_nodes[name]) == id(item)
                _nodes[name] = item
                _allow_node = False
                it = item.values()

            elif isinstance(item, dict):
                it = item.values()
            elif isinstance(item, list):
                it = item
            else:
                return
            for i in it:
                _check_cfg(i, allow_node=_allow_node)

        # 1. check configs to pick out node modules
        for v in module_dict.values():
            _check_cfg(v)

        # 2. build nodes and make them traceable
        nodes = {}
        for k, v in _nodes.items():
            _v = build_from_registry(v)
            GraphTracer.register_class_traceable_under_scope(_v.__class__)
            nodes[k] = _v

        return nodes

    def _register_nodes(self, node2name):
        def _register(node):
            if isinstance(node.op, nn.Module) and node.op in node2name:
                mod_name = node2name[node.op]
                self.add_module(mod_name, node.op)
                self._node_name_2_mod_name[node.name] = mod_name

            if callable(node.op) and not isinstance(node.op, nn.Module):
                # pass nn.Module to a traceable function
                for value in list(node.args) + list(node.kwargs.values()):
                    if isinstance(value, nn.Module) and value in node2name:
                        mod_name = node2name[value]
                        self.add_module(mod_name, value)
                        # TODO(min.du): update self._node_name_2_mod_name #
                        # when a traceable function with nn.Module as input
                        # node_name_2_mod_name is not right, because
                        # one node may have more than one mod.
                        self._node_name_2_mod_name[node.name] = mod_name

        # 2. register modules by traversing the traced graph
        self._graph.post_order_dfs_visit(fvisit=_register)

    def _check_params_and_buffers(self, module_dict):
        """Check parameters and buffers.

        Make sure all trainable variables in each task module
        are covered in graph model scope
        """

        # 1. collect all parameters and buffers under management
        _params_n_buffers = set()
        for p in self.parameters():
            _params_n_buffers.add(p)
        for b in self.buffers():
            _params_n_buffers.add(b)

        # 2. check
        for t, m in module_dict.items():
            for name, p in m.named_parameters():
                assert (
                    p in _params_n_buffers
                ), f"Parameter {name} of task {t} not covered"
            for name, b in m.named_buffers():
                assert (
                    b in _params_n_buffers
                ), f"Buffer {name} of task {t} not covered"
        return

    @staticmethod
    def _build_variables(
        inputs, opt_inputs, task_inputs, lazy_forward, device=None
    ):
        def _rec_to_device(item, device):
            if device is None:
                return item
            if isinstance(item, torch.Tensor):
                return item.to(device)
            elif isinstance(item, Sequence):
                return type(item)([_rec_to_device(i, device) for i in item])
            elif isinstance(item, Mapping):
                return type(item)(
                    {k: _rec_to_device(v, device) for k, v in item.items()}
                )

        def _build_variable(key, value):
            if not lazy_forward:
                assert (
                    value is not None
                ), "input data can not be None when `lazy_forward` is False"
            value = _rec_to_device(value, device)
            return Variable(name=key, data=value)

        def _build_opt_variable(key, value):
            if not lazy_forward:
                assert (
                    value is not None
                ), "input data can not be None when `lazy_forward` is False"
            value = _rec_to_device(value, device)
            return OptionalVariable(name=key, default=value)

        assert not set(inputs.keys()).intersection(set(opt_inputs.keys()))

        inputs_var = {k: _build_variable(k, v) for k, v in inputs.items()}
        inputs_var.update(
            {k: _build_opt_variable(k, v) for k, v in opt_inputs.items()}
        )

        all_inputs_var = {}
        for k, v in task_inputs.items():
            all_inputs_var[k] = _build_variable(k, {k: v})

        return inputs_var, all_inputs_var

    @staticmethod
    def _merge_nodes(heads):
        class _Node(object):
            def __init__(self, node: Node, *outputs):
                self._node = node
                self._outputs = list(outputs)

            def __hash__(self) -> int:
                return hash(self._node)

        _nodes = {}

        def _reverse_symbols(_heads):
            def fvisit(node: Node):
                _start_node.append(node)
                if node not in _nodes:
                    _nodes[node] = _Node(node)
                for n in node.inputs:
                    if n in _nodes:
                        _nodes[n]._outputs.append(_nodes[node])
                    else:
                        _nodes[n] = _Node(n, _nodes[node])

            start_node = None
            for h in _heads:
                _start_node = []
                h.post_order_dfs_visit(fvisit=fvisit)
                if start_node is not None:
                    assert start_node == _start_node[0]
                start_node = _start_node[0]
            return start_node

        start = _reverse_symbols(heads)

        _node_cnt = defaultdict(int)
        stack = [_nodes[start]]
        _visited = set()
        _abandoned = set()

        while stack:
            _node = stack.pop(0)
            if _node in _visited:
                continue

            _node_cnt[_node] += 1
            if not _node._node.inputs or (
                _node_cnt[_node] >= len(_node._node.inputs)
            ):
                _visited.add(_node)

            _op_args_to_node = {}
            new_outputs = []

            # n is one of the output nodes of current node
            for n in _node._outputs:
                c_op = n._node.op
                c_args = n._node.args

                # check hashable
                try:
                    hash(c_args)
                except TypeError:
                    c_args = flatten(c_args)[0]

                # should replace n._node.op with c_op
                if (c_op, *c_args) in _op_args_to_node:

                    for out_n in n._outputs:
                        # update args of outputs of n
                        out_n._node._args = tuple(
                            _op_args_to_node[(c_op, *c_args)]._node
                            if hasattr(arg, "op") and arg.op == c_op
                            else arg
                            for arg in out_n._node.args
                        )

                        # update inputs of outputs of n
                        new_inputs = {}
                        for input in out_n._node.inputs:
                            _args = input.args
                            try:
                                hash(_args)
                            except TypeError:
                                _args = flatten(_args)[0]

                            new_node = (
                                _op_args_to_node[(input.op, *_args)]._node
                                if input.op == c_op and input.args == c_args
                                else input
                            )

                            new_inputs[new_node] = new_node
                        out_n._node._inputs = new_inputs

                        # update outputs of c_op
                        if (
                            out_n
                            not in _op_args_to_node[(c_op, *c_args)]._outputs
                        ):  # noqa
                            _op_args_to_node[(c_op, *c_args)]._outputs.append(
                                out_n
                            )

                    else:
                        # node without outputs are leaves,
                        # can be abandoned directly
                        _abandoned.add(n._node)

                # new one
                else:
                    _op_args_to_node[(c_op, *c_args)] = n
                    new_outputs.append(n)
                    stack.append(n)

            _node._outputs = new_outputs

        # get rid of redundant outputs
        for h in heads:
            h._outputs = [
                node for node in h._outputs if node not in _abandoned
            ]

    def _transform_graph(self, func: Callable):
        func(self._graph)

    def _get_sub_graph(self, out_names: Union[str, Sequence[str]]) -> Symbol:
        """Select part of the graph outputs by `out_names` to get sub graph.

        Args:
            out_names: Names of graph outputs, should be a subset of
            `self._output_names` .

        Returns:
            :class:`capbc.workflow.symbol.Symbol`:
                A sub graph of `self._graph` .
        """
        assert self._cached_graphs, "build graph topology first"
        return group(
            *[self._cached_graphs[name] for name in _as_list(out_names)]
        )

    def forward(
        self,
        inputs: Dict[str, Any],
        out_names: Optional[Union[str, Sequence[str]]] = None,
    ) -> namedtuple:  # noqa: D205,D400
        r"""
        Provide graph output names and necessary input data to forward full or
        sub graph.

        Args:
            out_names: Graph output names, should be a subset of
                `self._output_names` , i.e. should keep accordance with
                the keys of `name2out` which is returned from
                `self.topology_builder` .

                If None, means to forward the whole graph.

                If not None, we will use it to get a sub graph then forward.

            inputs: A dict of (input name, data), should be a subset of
                `self.inputs` , providing necessary input data to forward the
                full or sub graph.

                .. note::

                    Only provide reliable inputs used in graph forward,
                    extra inputs will cause error.

        Returns:
            A namedtuple consists of output Tensors in order of `out_names` .

            .. note::

                Return dict is forbidden by torch.jit.trace(), so return
                namedtuple, or tuple (list is not recommended).

        Example::

            gm = GraphModel(nodes, inputs, topology_builder)
            inputs = dict(img=torch.randn((1, 3, 600, 800)))
            results = gm(inputs, out_names=['person', 'vehicle'])

            print(results._fields)
            # (
            #     'person_cls_pred_0_', 'person_cls_pred_1_', ... ,
            #     'vehicle_cls_pred_0_', 'vehicle_cls_pred_1_', ... ,
            #     ...
            # )

            results[0] == results.person_cls_pred_0_
            # True

            results[1] == results.person_cls_pred_1_
            # True

        """
        # 0. check
        assert self._graph is not None, "init graph first"
        assert isinstance(
            inputs, dict
        ), "MultitaskGraphModel inputs should be a dict but get %s" % type(
            inputs
        )

        if out_names is None:
            out_names = self._output_names
        else:
            out_names = tuple(_as_list(out_names))
            assert len(out_names) > 0
            for n in out_names:
                assert (
                    n in self._output_names
                ), "%s not in output names: %s" % (n, self._output_names)
            # sort names to get cached graph, executor, so that
            # ['name1', 'name2'], ['name2', 'name1'] share the same executor.
            out_names = sorted(out_names)

        # 1. make sure it's not in DataParallel mode
        assert not getattr(
            self, "_is_replica", False
        ), "Don't use DataParallel"

        # 2. get executor
        sort_names = out_names
        tag = ">".join(sort_names)
        if tag not in self._cached_execs:
            if tag not in self._cached_graphs:
                self._cached_graphs[tag] = self._get_sub_graph(sort_names)

            sub_graph = self._cached_graphs[tag]
            self._cached_execs[tag] = SymbolExecutor(sub_graph)

        # 3. forward

        # file_index = inputs["img_metas_batch"][0][0].split("/")[-1].split(".")[0]
        # input_save = np.array(inputs["img"].cpu()).flatten()
        # np.savetxt("bev_input/input_" + str(file_index) + ".txt",input_save,fmt="%.8f")

        results = self._cached_execs[tag](inputs)

        # restore the original data structure
        new_results = OrderedDict()
        if len(sort_names) == 1:
            name = sort_names[0]
            _, fmts = self._name2fmts[name]
            grouped, rest = regroup(results, fmts)
            if len(rest) or not isinstance(fmts, Iterable):
                new_results[name] = results
            else:
                new_results[name] = grouped
        else:
            for name in sort_names:
                (start, end), fmts = self._name2fmts[name]
                new_results[name] = regroup(results[start:end], fmts)[0]

        results = new_results

        assert len(results) == len(sort_names), "%d vs. %d" % (
            len(results),
            len(sort_names),
        )
        if self.flatten_outputs:
            # 3. reorder results in order of out_names
            outs = []
            for name, res in results.items():

                name2out = to_flat_ordered_dict(
                    res,
                    key_prefix=name,
                    flat_condition=self._output_names2flat_conditions[name],
                )

                for k in name2out:
                    if isinstance(name2out[k], list):
                        name2out[k] = tuple(name2out[k])

                # torch.jit.trace() recommend us to convert dict to namedtuple
                try:
                    OrderedOutput = namedtuple(
                        "OrderedOutput", name2out.keys()
                    )
                except SyntaxError as e:
                    logger.error("name2out keys: {name2out.keys()}")
                    raise e
                outs.append(OrderedOutput(**name2out))

            return tuple(outs)
        else:
            return results

    def fuse_model(self):
        for module in self.children():
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from cap.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

    def set_calibration_qconfig(self):
        import changan_plugin_pytorch as changan

        self.qconfig = changan.quantization.get_default_calib_qconfig()

        for module in self.children():
            if module is not None:
                if hasattr(module, "set_calibration_qconfig"):
                    module.set_calibration_qconfig()

    @property
    def output_names(self):
        """Names of graph output variables."""
        return self._output_names

    @property
    def graph(self):
        """Full graph which represents GraphModel's computational topology."""  # noqa: E501
        return self._graph

    def _named_modules_by_outname(
        self, out_names: Tuple[str], iter_fn: Callable, prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get named modules contained by the sub graph of output names."""
        sub_graph = self._get_sub_graph(out_names)
        _modules = {}

        def _get_modules(node):
            if isinstance(node.op, nn.Module):
                mod_name = self._node_name_2_mod_name[node.name]
                _modules[mod_name] = self._modules[mod_name]

        sub_graph.post_order_dfs_visit(_get_modules)

        memo = set()
        for n, m in _modules.items():
            iters = iter_fn(m)
            for k, v in iters:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = prefix + n + ("." if n else "") + k
                yield name, v

    def named_parameters_by_outname(
        self, out_names: Tuple[str], prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get all named parameters that contained by sub-graph of outname."""
        gen = self._named_modules_by_outname(
            out_names, lambda m: m.named_parameters(), prefix
        )
        return gen

    def named_buffers_by_outname(
        self, out_names: Tuple[str], prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get all named buffers that contained by sub-graph of outname."""
        gen = self._named_modules_by_outname(
            out_names, lambda m: m.named_buffers(), prefix
        )
        return gen

    def named_modules_by_outname(
        self, out_names: Tuple[str], prefix: str = ""
    ) -> Tuple[str, Any]:
        """Get all named modules that contained by sub-graph of outname."""
        gen = self._named_modules_by_outname(
            out_names, lambda m: m.named_modules(), prefix
        )
        return gen
