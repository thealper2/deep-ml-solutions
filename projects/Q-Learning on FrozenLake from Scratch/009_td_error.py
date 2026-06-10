def td_error(target, q_table, state, action):
    return target - q_table[state, action]
