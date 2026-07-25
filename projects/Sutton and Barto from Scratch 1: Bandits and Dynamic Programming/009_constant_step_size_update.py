def constant_step_size_update(q_values, action, reward, alpha):
    new_q_values = q_values.copy()
    new_q_values[action] += alpha * (reward - new_q_values[action])
    return new_q_values
