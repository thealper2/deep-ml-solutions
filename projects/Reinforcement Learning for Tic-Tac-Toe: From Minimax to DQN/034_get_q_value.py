def get_q_value(q_table, state_key, action):
    return q_table.get((state_key, action), 0.0)
