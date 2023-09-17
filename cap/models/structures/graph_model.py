# Copyright (c) Changan Auto. All rights reserved.
import copy
import inspect
import logging
import threading
from collections import OrderedDict, namedtuple
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from capbc.workflow.engine import SymbolExecutor
from capbc.workflow.proxy import Variable, WorkflowVariable, get_traced_graph
from capbc.workflow.symbol import Symbol
from capbc.workflow.trace import GraphTracer, is_workflow_traceable

from cap.registry import OBJECT_REGISTRY
from cap.utils.apply_func import (
    _as_list,
    is_list_of_type,
    to_flat_ordered_dict,
)

__all__ = ["GraphModel"]

logger = logging.getLogger(__name__)

GraphTracer.register_basic_types(torch.Tensor)


@OBJECT_REGISTRY.register
class GraphModel(nn.Module):
    r"""
    A Module contains a graph to represent computational topology.

    Each graph consist of multiple nodes, and each node wraps an operator. We
    make `nn.Module` instance, e.g. backbone, neck, head, as one of the graph
    nodes.

    The purpose we introduce a graph object:
        In some cases, we just want to forward part (not all) of the task
        specific heads of a multitask model, which may have some parameters
        shared by multiple tasks.

    2 steps to accomplish this purpose:
         (1) Get a sub graph by specific a subset of graph output nodes.
         (2) Forward it, which means to forward part of the GraphModel.

    An example:

    .. code-block:: python

            # make nn.Module traceable
            GraphTracer.register_class_traceable_under_scope(torch.nn.Module)
            GraphTracer.register_basic_types(torch.Tensor)

            def build_topo(nodes, inputs):
                #
                # Computational Topology:
                #
                # (img) -> backbone -> neck -> head1 -> (task1_out)
                #                         \ -> head2 -> (task2_out)
                #                         \ -> head3 -> (task3_out)
                #
                name2out = OrderedDict()
                bone_feats = nodes['backbone'](inputs['img'])
                neck_feats = nodes['neck'](bone_feats)
                name2out['task1_out'] = nodes['head1'](neck_feats)
                name2out['task2_out'] = nodes['head2'](neck_feats)
                name2out['task3_out'] = nodes['head3'](neck_feats)
                return name2out

            gm = GraphModel(
                nodes=dict(
                    backbone=ResNet(...),
                    neck=FPN(...),
                    head1=FCOSHead(...),
                    head2=FCOSHead(...),
                    head3=FCOSHead(...),
                ),
                inputs=dict(img=None),
                topology_builder=build_topo,
                lazy_forward=True,
            )

            batch = dict(img=torch.randn((1, 3, 600, 800)))

            # Case1, forward the whole model (full graph). Topology:
            #
            # (img) -> backbone -> neck -> head1 -> (task1_out)
            #                         \ -> head2 -> (task2_out)
            #                         \ -> head3 -> (task3_out)
            #
            gm(batch, out_names=None)

            # Case2, forward task1 (sub graph). Topology:
            #
            # (img) -> backbone -> neck -> head1 -> (task1_out)
            #
            gm(batch, out_names='task1_out')

            # Case3, forward task2 and task3 (sub graph). Topology:
            #
            # (img) -> backbone -> neck -> head2 -> (task2_out)
            #                         \ -> head3 -> (task3_out)
            #
            gm(batch, out_names=['task2_out', 'task3_out'])

    Args:
        nodes: Graph nodes, can be any callable as well as traceable
            (by `workflow.GraphTracer`) object.
        inputs: {input name: value} pairs, used to build the graph's input
            variables, i.e. placeholders. Value can be None (unset) when
            `lazy_forward` is True.
        topology_builder: A callable func, with already build `nodes` and
            `inputs` as input arguments, used to build computational topology
            under `with GraphTracer()` scope.

            It returns an OrderedDict object to specific the ordered graph
            outputs (a collection of `WorkflowVariable` objects, and the
            `WorkflowVariable.idata` attr contains model output Tensors).

            This func will be called only once to get traced graph before
            formal training, i.e. the traced graph is a `static graph`.
        lazy_forward: Whether forward graph nodes when topology building.
            If True, means not immediately forward, delay to
            :func:`GraphModel.forward()` time when topology building is done.
            At this case, `inputs` can be (str, None) pairs, i.e. you don't
            have to provide input values, just provide input names to create
            placeholders.

            If False, means immediately forward, i.e. forward to build a
            graph, then you achieve node outputs immediately, of course, you
            have to provide input values. This is useful for debugging the
            topology before formal training.

            .. note::

                No matter True or False, finally a static traced graph is
                build, i.e. will not change among training.

            Refer :attr:`GraphTracer.imperative` for more.
        flat_output_condition: Function with `obj` as input, return
            `True/False` means whether flat this `obj` or not, used when
            flatting graph output results. See `to_flat_ordered_dict` for more.
            Deprecated in future.
        output_names2flat_conditions: Dict of function with `values` or
            (`values`, `key`) as input, return `True/False` means whether flat
            this `values` or not, used when flatting graph output results.
            See `to_flat_ordered_dict` for more.

    """

    # GraphTracer is not threading safe #
    TRACER_LOCK = threading.Lock()

    def __init__(
        self,
        nodes: Dict[str, Union[dict, nn.Module, Callable]],
        inputs: Dict[str, Any],
        topology_builder: Callable[
            [Dict[str, Callable], Dict[str, Variable]],
            Dict[str, WorkflowVariable],
        ],
        lazy_forward: Optional[bool] = True,
        flat_output_condition: Optional[Callable[[Any], bool]] = None,
        output_names2flat_conditions: Dict[
            str, Callable[[Any], bool]
        ] = None,  # noqa
    ):
        super(GraphModel, self).__init__()
        assert callable(topology_builder)

        self.nodes = nodes
        self.inputs = inputs
        self.topology_builder = topology_builder
        self.lazy_forward = lazy_forward

        if output_names2flat_conditions is None:
            self._output_names2flat_conditions = {}
        else:
            assert isinstance(output_names2flat_conditions, dict)
            self._output_names2flat_conditions = output_names2flat_conditions

        if flat_output_condition is not None:
            logger.warning(
                "`flat_output_condition` is deprecated, "
                "please change to `output_names2flat_conditions`"
            )
            assert callable(flat_output_condition)
            self._flat_condition = flat_output_condition
        else:
            self._flat_condition = None

        self._graph = None
        self._name2output = None
        self._output_names = None
        self._cached_graphs = {}
        self._cached_execs = {}
        self._node_name_2_mod_name = {}
        # device id 2 replicate info, only useful when GraphModel is warped by
        # DataParallel (implement by threading). Set default None to strictly
        # ensure dict is threading safe. Assume max num of GPU in one machine
        # is 8 (even more is ok).
        self._device_2_rep_info = {i: None for i in range(8)}

        self._build_graph()

        # set default flat_conditions
        for name in self._output_names:
            if name not in self._output_names2flat_conditions:
                self._output_names2flat_conditions[
                    name
                ] = lambda k, v: not is_list_of_type(v, torch.Tensor)

    def _build_nodes(
        self, clone_node: bool = False
    ) -> Dict[str, Any]:  # noqa: D205,D400
        """Instantiate nodes, e.g. backbone, neck, and other traceable
        objects.

        Args:
            clone_node: When node in `nodes` is already instantiated, whether
                deepcopy it. Set True is much safer when nodes will be used by
                multiple threads. Default False.

        Returns::
            nodes: Instantiated node objects.
        """
        nodes = {}
        for k, v in self.nodes.items():
            if clone_node:
                v = copy.deepcopy(v)
                v = v.cpu()

            assert is_workflow_traceable(v), (
                "Please make %s workflow traceable by `@make_traceable` "
                "or `GraphTracer.register_class_traceable_under_scope`"
                % (v if inspect.isfunction(v) else type(v))
            )
            nodes[k] = v

        return nodes

    def _build_variables(
        self, clone_value: bool = False
    ) -> Dict[str, Variable]:  # noqa: D205,D400
        """Wrap each input data as an `Variable` object, so that they can be
        traced by `workflow.GraphTracer`.

        Args:
            clone_value: Whether clone input values. Set True is much safer
                when inputs will be used by multiple threads.

        Returns::
            variables: `workflow.GraphTracer.Variable` objects, which are graph
                input nodes, i.e. input placeholders.

        """
        variables = {}
        for k, v in self.inputs.items():
            if v is not None and clone_value:
                v = copy.deepcopy(v)
            if not self.lazy_forward:
                assert (
                    v is not None
                ), "input data can not be None when `lazy_forward` is False"
            variables[k] = Variable(name=k, data=v)

        return variables

    def _build_graph(self, clone: bool = False):  # noqa: D205,D400
        """Build graph using `self.nodes` and `self.inputs`, to represent
        computational topology.

         It does the following:

        (1) Instantiate the graph components: Nodes (nodes) and Variables
            (inputs).
        (2) Pass these components to topology_builder to construct the graph's
            computational topology and achieve graph outputs (a collection of
            `WorkflowVariable` objects, and `WorkflowVariable.idata` attr
            contains model output Tensors).
        (3) Get a traced graph using these output variables.
        (4) Add those `nn.Module` type nodes to `self._modules` , so that
            GraphModel can be traced by torch.jit.trace().

        Args:
            clone: Whether clone node instance or input value when building
                graph nodes and variables.
                Refer function `_build_nodes` and `_build_variables`
        """
        # 1. build
        # synchronously execute GraphTracer, cos it is not threading safe.
        # let nn.Module can be traced by GraphTracer
        GraphTracer.register_class_traceable_under_scope(torch.nn.Module)

        with self.TRACER_LOCK, GraphTracer(imperative=not self.lazy_forward):
            nodes = self._build_nodes(clone_node=clone)
            inputs = self._build_variables(clone_value=clone)

            name2out = self.topology_builder(nodes, inputs)

            assert isinstance(name2out, OrderedDict), (
                "Please use `OrderedDict` to represent the ordered graph "
                "outputs, but get %s" % type(name2out)
            )

            for _k, v in name2out.items():
                for obj in _as_list(v):
                    assert isinstance(obj, WorkflowVariable), (
                        "graph outputs should be one or a sequence of "
                        "`WorkflowVariable` objects, but get %s" % type(obj)
                    )

            self._name2output = name2out
            self._output_names = tuple(self._name2output.keys())

            # get graph, a Symbol
            self._graph = self._get_sub_graph(self._output_names)

        node2name = {v: k for k, v in nodes.items()}

        def _register(node):
            if isinstance(node.op, nn.Module) and node.op in node2name:
                mod_name = node2name[node.op]
                self.add_module(mod_name, node.op)
                self._node_name_2_mod_name[node.name] = mod_name

            if callable(node.op) and not isinstance(node.op, nn.Module):
                # pass nn.Module to a traceable function
                mod_list = []
                for value in list(node.args) + list(node.kwargs.values()):
                    if isinstance(value, nn.Module) and value in node2name:
                        mod_name = node2name[value]
                        self.add_module(mod_name, value)
                        # TODO(min.du): update self._node_name_2_mod_name #
                        # when a traceable function with nn.Module as input
                        # node_name_2_mod_name is not right, because
                        # one node may have more than one mod.
                        mod_list.append(mod_name)
                self._node_name_2_mod_name[node.name] = mod_list

        # 2. register
        self._graph.post_order_dfs_visit(fvisit=_register)

    def _get_sub_graph(self, out_names: Union[str, Sequence[str]]) -> Symbol:
        """Select part of the graph outputs by `out_names` to get sub graph.

        Args:
            out_names: Names of graph outputs, should be a subset of
            `self._output_names` .

        Returns:
            :class:`capbc.workflow.symbol.Symbol`:
                A sub graph of `self._graph` .
        """
        assert self._name2output is not None, "build graph topology first"
        out_names = _as_list(out_names)
        sub_outs = []
        for n in out_names:
            sub_outs += _as_list(self._name2output[n])

        return get_traced_graph(sub_outs)

    def _prepare_for_data_parallel(
        self, inputs: Dict[str, Any]
    ):  # noqa: D205,D400
        """Replicate graph and cache it only for
        `torch.nn.parallel.DataParallel` training mode.

        It does the following:

        (1) Get device info using one of the `torch.Tensor` objects in
        `inputs`.
        (2) Replicate graph (or use pre step cache graph) , i.e. create
        independent graph object for each thread (`DataParallel` creates multi
        threads to do training in parallel).
        (3) Binding new modules (replicated by DataParallel) to the new graph.

        Args:
            inputs: GraphModel forward inputs.
        """
        # 1. get device
        device = None
        for _k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if device is None:
                    device = v.device
                else:
                    assert device == v.device, (
                        f"{device} vs. {v.device}, "
                        f"`inputs` contains tensors locate at different "
                        f"device, we can't decide it."
                    )

        if device is None:
            raise AttributeError(
                "We can't get device info because `inputs` "
                "do not contain any torch.Tensor object."
            )

        assert device.type == "cuda", (
            f"You are using DataParallel training,"
            f" device type should be `cuda`, but get {device.type}."
        )

        # 2. build graph for current thread and replace nn modules using new
        # ones
        new_mods = {}
        for k in self._node_name_2_mod_name.values():
            if not isinstance(k, (list, tuple)):
                k = [k]
            for m in k:
                if getattr(self, m, None):
                    # use already build (by DataParallel) replicate module,
                    # not to create.
                    new_mods[m] = getattr(self, m)

        def _replace_mod_op(node):
            if node.name in self._node_name_2_mod_name:
                mod_name = self._node_name_2_mod_name[node.name]
                if isinstance(mod_name, str):
                    node.op = new_mods[mod_name]
                else:
                    # node is a tracable func, do not need to replace
                    pass

        device_id = device.index
        if (
            device_id not in self._device_2_rep_info
            or self._device_2_rep_info[device_id] is None
        ):
            # trace to rebuild graph
            self._build_graph(clone=True)
            self._graph.post_order_dfs_visit(fvisit=_replace_mod_op)

            # clear cache to renew
            self._cached_graphs = {}
            self._cached_execs = {}
            self._node_name_2_mod_name = {}

            # cache for next time forward
            self._device_2_rep_info[device_id] = {
                "_graph": self._graph,
                "_name2output": self._name2output,
                "_cached_graphs": self._cached_graphs,
                "_cached_execs": self._cached_execs,
                "_node_name_2_mod_name": self._node_name_2_mod_name,
            }

        else:
            # Use cache to init graph, not to trace to rebuild it because
            # rebuild is synchronized and slow.
            rep_info = self._device_2_rep_info[device_id]

            # replace graph op using new ones
            self._graph = rep_info["_graph"]
            self._graph.post_order_dfs_visit(fvisit=_replace_mod_op)

            # use other cache
            self._name2output = rep_info["_name2output"]
            self._cached_graphs = rep_info["_cached_graphs"]
            self._cached_execs = rep_info["_cached_execs"]
            self._node_name_2_mod_name = rep_info["_node_name_2_mod_name"]

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
                the keys of `self._name2output` which is return from
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
        ), "GraphModel inputs should be a dict but get %s" % type(inputs)
        for k, _ in inputs.items():
            assert k in self.inputs, "%s not in input names: %s" % (
                k,
                self.inputs.keys(),
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

        # 1. prepare only for DataParallel training
        if getattr(self, "_is_replica", False):
            self._prepare_for_data_parallel(inputs)

        # 2. get executor
        # sort names to get cached graph, executor, so that
        # ['name1', 'name2'] and ['name2', 'name1'] share the same executor.
        sort_names = sorted(out_names)
        tag = ">".join(sort_names)
        if tag not in self._cached_execs:
            if tag not in self._cached_graphs:
                self._cached_graphs[tag] = self._get_sub_graph(sort_names)

            sub_graph = self._cached_graphs[tag]
            self._cached_execs[tag] = SymbolExecutor(sub_graph)

        # 3. forward
        results = self._cached_execs[tag](inputs)
        if len(sort_names) == 1:
            results = [results]
        assert len(results) == len(sort_names), "%d vs. %d" % (
            len(results),
            len(sort_names),
        )

        # 3. reorder results in order of out_names
        name2res = dict(zip(sort_names, results))
        order_res = [name2res[n] for n in out_names]
        outs = []
        for name, res in zip(out_names, order_res):
            # legacy: compatible with flat_output_condition
            if self._flat_condition and not self._output_names2flat_conditions:
                self._output_names2flat_conditions[name] = self._flat_condition

            name2out = to_flat_ordered_dict(
                res,
                key_prefix=name,
                flat_condition=self._output_names2flat_conditions[name],
            )

            # torch.jit.trace() recommend us to convert dict to namedtuple
            try:
                OrderedOutput = namedtuple("OrderedOutput", name2out.keys())
            except SyntaxError as e:
                logger.error("name2out keys: {name2out.keys()}")
                raise e
            outs.append(OrderedOutput(**name2out))

        return tuple(outs)

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
            elif callable(node.op) and not isinstance(node.op, nn.Module):
                # node is tracable func, get all related modules
                # mod_name is a list of module names
                mod_name = self._node_name_2_mod_name[node.name]
                for m in mod_name:
                    _modules[m] = self._modules[m]

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
