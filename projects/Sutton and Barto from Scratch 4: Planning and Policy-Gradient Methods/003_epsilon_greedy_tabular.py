import numpy as np


def epsilon_greedy_tabular(q_table, state, epsilon, rng):
    values = np.asarray(q_table[state])
    n_actions = values.shape[-1]

    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))

    best = np.max(values)
    candidates = np.flatnonzero(values == best)
    return int(rng.choice(candidates))