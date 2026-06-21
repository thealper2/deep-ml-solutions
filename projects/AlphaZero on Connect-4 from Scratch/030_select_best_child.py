def select_best_child(node, legal_actions, c_puct=1.5):
    best_action = None
    best_child = None
    best_score = float('-inf')

    for action in legal_actions:
        child = parent['children'][action]
        score = ucb_score(parent, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
