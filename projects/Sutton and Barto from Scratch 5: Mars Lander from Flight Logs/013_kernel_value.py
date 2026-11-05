import numpy as np


def kernel_value(query_states, memory_states, memory_targets, bandwidth):
    Q = unit_states(query_states)
    M = unit_states(memory_states)
    g = np.asarray(memory_targets, dtype=float)

    d2 = np.sum((Q[:, None, :] - M[None, :, :]) ** 2, axis=2)
    d2 = d2 - d2.min(axis=1, keepdims=True)
    k = np.exp(-d2 / (2.0 * bandwidth ** 2))
    return (k @ g) / k.sum(axis=1)