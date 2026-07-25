def sample_average_update(q_values, action_counts, action, reward):
    new_q_values = q_values.copy()
    new_action_counts = action_counts.copy()
    
    new_action_counts[action] += 1
    n = new_action_counts[action]

    new_q_values[action] += (reward - new_q_values[action]) / n
    return new_q_values, new_action_counts
