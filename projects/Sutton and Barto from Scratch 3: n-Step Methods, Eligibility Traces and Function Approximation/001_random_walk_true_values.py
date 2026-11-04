import numpy as np

def random_walk_true_values(n_states):
    i = np.arange(n_states, dtype=float)
    return -1.0 + 2.0 * (i + 1.0) / (n_states + 1.0)