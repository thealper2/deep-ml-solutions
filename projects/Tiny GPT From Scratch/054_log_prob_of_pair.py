def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    prob = index_element(p_matrix, current_id, next_id)
    return array_log(prob).item()
