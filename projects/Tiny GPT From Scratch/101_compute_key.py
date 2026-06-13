def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    return np.dot(x, w_k)
