def select_leaf(root, c_puct):
    node = root
    while node.get('is_expanded', False):
        legal_actions = list(node['children'].keys())
        action, child = select_best_child(node, legal_actions, c_puct)
        node = child

    return node
