def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    db = np.sum(dy, axis=0)
    return db
