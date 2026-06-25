def q_learning_nonterminal_target(reward, gamma, q_table, next_state_key, next_legal_actions):
    """Return the TD target r + gamma * max_a' Q(s', a') over legal next actions."""
    if not next_legal_actions:
        return reward
        
    max_q = max(get_q_value(q_table, next_state_key, a) for a in next_legal_actions)
    return reward + gamma * max_q
