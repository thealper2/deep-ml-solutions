import numpy as np

def epsilon_soft_action(q_values, epsilon, rng):
    u = rng.random()

    if u < epsilon:
        return int(rng.integers(0, len(q_values)))

    return int(np.argmax(q_values))