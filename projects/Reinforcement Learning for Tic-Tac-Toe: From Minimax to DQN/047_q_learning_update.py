def q_learning_update(q_table, state_key, action, target, alpha):
    """Apply Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a)) and return the new value."""
    new_value = get_q_value(q_table, state_key, action) + alpha * (target - get_q_value(q_table, state_key, action))
    set_q_value(q_table, state_key, action, new_value)
    return new_value
