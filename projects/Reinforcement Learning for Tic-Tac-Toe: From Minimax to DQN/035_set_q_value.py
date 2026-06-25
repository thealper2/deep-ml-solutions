def set_q_value(q_table, state_key, action, value):
    """Write a new Q-value for a (state, action) pair into the Q-table."""
    q_table[(state_key, action)] = value
