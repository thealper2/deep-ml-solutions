def sarsa_on_policy_update(q_table, state_key, action, reward, next_state_key, next_action, done, alpha, gamma):
    """Apply one on-policy SARSA update and return the updated q_table."""
    if done:
        target = reward
    else:
        next_q = get_q_value(q_table, next_state_key, next_action)
        target = reward + gamma * next_q

    current_q = get_q_value(q_table, state_key, action)
    new_q = current_q + alpha * (target - current_q)
    set_q_value(q_table, state_key, action, new_q)
    
    return q_table
