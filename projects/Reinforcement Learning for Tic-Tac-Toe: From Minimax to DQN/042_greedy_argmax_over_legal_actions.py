def greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng):
    """Return the legal action with the highest Q-value (random tie-break)."""
    best_value = float('-inf')
    best_actions = []

    for action in legal_actions:
        value = get_q_value(q_table, state_key, action)
        if value > best_value:
            best_value = value
            best_actions = [action]
        elif value == best_value:
            best_actions.append(action)

    idx = rng.integers(0, len(best_actions))
    return best_actions[idx]
