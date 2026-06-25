def epsilon_greedy_select_action(q_table, state_key, legal_actions, epsilon, rng):
    """Choose an action via epsilon-greedy over the legal actions."""
    if rng.random() < epsilon:
        return epsilon_greedy_explore_move(legal_actions, rng)
    else:
        best_action = legal_actions[0]
        best_value = get_q_value(q_table, state_key, best_action)
        for action in legal_actions[1:]:
            value = get_q_value(q_table, state_key, action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
