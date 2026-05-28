import numpy as np

def activation_patching(clean_acts, corrupted_acts, weights):
    """
    Compute per-location activation-patching recovery scores.

    Args:
        clean_acts: 2D array-like of shape (L, P)
        corrupted_acts: 2D array-like of shape (L, P)
        weights: 2D array-like of shape (L, P)

    Returns:
        A 2D Python list of shape (L, P) of recovery scores.
    """
    clean_acts = np.array(clean_acts)
    corrupted_acts = np.array(corrupted_acts)
    weights = np.array(weights)

    f = lambda A: np.sum(weights * (A ** 2))

    f_clean = f(clean_acts)
    f_corrupted = f(corrupted_acts)

    denominator = f_clean - f_corrupted

    if denominator == 0:
        return np.zeros_like(clean_acts).tolist()

    diff_sq = (clean_acts ** 2) - (corrupted_acts ** 2)
    numerators = weights * diff_sq
    recovery_scores = numerators / denominator
    return recovery_scores.tolist()