import numpy as np


def corridor_policy(theta):
    t = np.asarray(theta, dtype=float)
    shifted = t - np.max(t)
    e = np.exp(shifted)
    return e / np.sum(e)