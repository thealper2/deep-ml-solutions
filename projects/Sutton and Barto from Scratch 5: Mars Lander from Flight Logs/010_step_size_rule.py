import numpy as np


def step_size_rule(X, tau):
    X = np.asarray(X, dtype=float)
    mean_sq = np.mean(np.sum(X * X, axis=1))
    return float(1.0 / (tau * mean_sq))