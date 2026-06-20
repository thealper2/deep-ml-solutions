def node_q_value(node):
    if node['visit_count'] == 0:
        return 0.0

    return node['value_sum'] / node['visit_count']
