import numpy as np

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    if len(labels) == 0:
        return 0.0

    counts = np.bincount(labels)
    probs = counts / len(labels)
    return 1.0 - np.sum(probs ** 2)
