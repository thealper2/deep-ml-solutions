def td_target(reward, gamma, q_table, next_state, done):
    if done:
        return float(reward)
    else:
        return reward + gamma * max_q_value(q_table, next_state)
