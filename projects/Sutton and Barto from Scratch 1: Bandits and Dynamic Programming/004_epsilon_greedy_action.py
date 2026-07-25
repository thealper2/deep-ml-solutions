def epsilon_greedy_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, len(q_values))
    else:
        return np.argmax(q_values)
