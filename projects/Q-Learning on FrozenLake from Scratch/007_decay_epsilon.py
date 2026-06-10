def decay_epsilon(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)
