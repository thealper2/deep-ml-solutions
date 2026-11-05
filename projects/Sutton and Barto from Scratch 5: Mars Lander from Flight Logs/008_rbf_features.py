import numpy as np


def rbf_features(states, centers, sigma):
    S = unit_states(states)
    C = np.asarray(centers, dtype=float).reshape(-1, 3)
    d2 = np.sum((S[:, None, :] - C[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * sigma ** 2))