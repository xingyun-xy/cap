import os
import inspect
import tempfile

try:
    import graphviz
except ImportError:
    graphviz = None

from capbc.utils import get_operator_name
from capbc.workflow.symbol import Symbol


__all__ = ['save_viz_graph', 'visualize_graph']


def save_viz_graph(graph, path):
    """
    Save visualized graph.

    Parameters
    ----------
    graph : :py:class:`graphviz.Digraph`
        Visualized graph
    path : str
        Render path.
    """
    assert graphviz is not None, 'Please install graphviz first'
    assert isinstance(graph, graphviz.Digraph), \
        f'Expected {graphviz.Digraph}, but get {str(type(graph))}'
    output_prefix, ext = os.path.splitext(path)
    graph.format = ext[1:]
    graph.render(output_prefix, cleanup=True)


def visualize_graph(graph, variable_node_color="red",
                    optional_variable_node_color="orange",
                    constant_variable_node_color="pink",
                    op_node_color="blue",
                    output_node_color="greenyellow",
                    render_op_module_name=True,
                    edge_color='black', save_path=None,
                    view=False):
    """
    Visualize :py:class:`Symbol`.

    Parameters
    ----------
    graph : :py:class:`Symbol`
        Symbolic graph
    variable_node_color : str, optional
        :py:class:`Variable` node color, by default "red"
    optional_variable_node_color : str, optional
        :py:class:`OptionalVariable` node color, by default "orange"
    constant_variable_node_color : str, optional
        :py:class:`Constant` node color, by default "pink"
    op_node_color : str, optional
        Operator node color, by default "blue"
    output_node_color : str, optional
        Output node color, by default "green"
    edge_color : str, optional
        Edge color, by default "black"
    save_path : str, optional
        Save visualize graph path, by default None
    view : bool, optional
        Whether visualize using matplotlib, by default False.
    """
    assert graphviz is not None, 'Please install graphviz first'

    assert isinstance(graph, Symbol), f'Expected type {Symbol}, but get {str(type(graph))}'  # noqa

    dot = graphviz.Digraph()

    def _is_input_node(node):
        return node.is_placeholder or node.is_optional_placeholder or node.is_constant  # noqa

    def _render_graph(node):

        if _is_input_node(node) and node in graph._outputs:
            render_args = dict(style="filled", fillcolor=output_node_color)
        else:
            render_args = dict()

        if node.is_placeholder:
            dot.node(node.name, color=variable_node_color, **render_args)
        elif node.is_optional_placeholder:
            dot.node(node.name, color=optional_variable_node_color,
                     **render_args)
        elif node.is_constant:
            dot.node(node.name, color=constant_variable_node_color,
                     **render_args)
        else:
            if node in graph._outputs:
                color = output_node_color
            else:
                color = op_node_color
            dot.node(node.name, label=f'{node.name}\nop={get_operator_name(node.op, render_op_module_name)}',  # noqa
                     color=color)

            real_inputs = set()

            def _fn(x):
                real_inputs.add(x.name)

            node._map_args(node.args, _fn)
            node._map_args(node.kwargs, _fn)

            for value in node._inputs.values():

                if value.name in real_inputs:
                    kwargs = dict()
                else:
                    kwargs = dict(style='dashed')  # dashed line for wait node

                dot.edge(value.name, node.name, color=edge_color, **kwargs)

    graph.post_order_dfs_visit(_render_graph)

    if save_path is not None:
        save_viz_graph(dot, save_path)

    if view:

        if save_path is None:
            save_path = tempfile.NamedTemporaryFile(suffix=".png").name
            save_viz_graph(dot, save_path)
            cleanup = True
        else:
            cleanup = False

        import cv2
        from matplotlib import pyplot as plt

        plt.imshow(cv2.imread(save_path)[:, :, ::-1])

        if cleanup:
            os.remove(save_path)

    return dot
