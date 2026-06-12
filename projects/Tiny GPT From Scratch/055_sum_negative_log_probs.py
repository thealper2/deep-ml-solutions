def sum_negative_log_probs(p_matrix, data):
    total = 0.0
    for i in range(len(data) - 1):
        log_prob = log_prob_of_pair(p_matrix, data[i], data[i + 1])
        total += -log_prob

    return total
