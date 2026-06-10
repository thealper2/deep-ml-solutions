def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    q_table[state, action] += alpha *  error
    return float(q_table[state, action])
