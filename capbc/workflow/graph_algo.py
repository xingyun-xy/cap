from capbc.utils import _as_list


__all__ = ['post_order_dfs_visit']


def post_order_dfs_visit(heads, visit_fn, indegree_fn, get_input_fn,
                         hash_fn=None, allow_circle=False):
    assert callable(visit_fn), 'visit_fn should be callable'
    assert callable(indegree_fn), 'indegree_fn should be callable'
    assert callable(get_input_fn), 'get_input_fn should be callable'
    if hash_fn is None:
        hash_fn = hash
    assert callable(hash_fn)

    visited_nodes = set()
    in_stack_nodes = set()
    stack = []

    for head_node_i in _as_list(heads):
        head_node_hash_value = hash_fn(head_node_i)
        if head_node_hash_value not in visited_nodes:
            stack.append({'node': head_node_i, 'num_visited': 0})
            visited_nodes.add(head_node_hash_value)
            in_stack_nodes.add(head_node_hash_value)
        while len(stack) > 0:
            last = stack[-1]
            if last['num_visited'] == indegree_fn(last['node']):
                visit_fn(last['node'])
                stack.pop()
                in_stack_nodes.remove(hash_fn(last['node']))
            else:
                input_node = get_input_fn(last['node'], last['num_visited'])
                last['num_visited'] += 1
                input_node_hash_value = hash_fn(input_node)
                if not allow_circle:
                    assert input_node_hash_value not in in_stack_nodes, \
                        f'Detect circle link at node {input_node}'
                if input_node_hash_value not in visited_nodes:
                    stack.append({'node': input_node, 'num_visited': 0})
                    visited_nodes.add(input_node_hash_value)
                    in_stack_nodes.add(input_node_hash_value)
