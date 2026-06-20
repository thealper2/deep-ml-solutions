def make_mcts_node(prior=0.0, parent=None):
    return {
        'prior': prior,
        'visit_count': 0,
        'value_sum': 0.0,
        'children': {},
        'parent': parent,
    }
