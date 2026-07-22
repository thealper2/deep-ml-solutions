import numpy as np

def weighted_index_score(Q, K, W):
    """
    Compute weighted multi-head index scores.

    Args:
        Q: array-like of shape (T, H, D)
        K: array-like of shape (S, D)
        W: array-like of shape (T, H)

    Returns:
        Nested list of shape (T, S) with index scores.
    """
    dot_products = np.einsum('thd,sd->ths', Q, K)
    relu_activated = np.maximum(0, dot_products)
    scores = np.einsum('th,ths->ts', W, relu_activated)
    return scores.tolist()
