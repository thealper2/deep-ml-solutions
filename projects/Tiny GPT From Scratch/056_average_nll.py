def average_nll(p_matrix, data):
    total_nll = sum_negative_log_probs(p_matrix, data)
    num_bigrams = len(data) - 1
    return total_nll / num_bigrams
