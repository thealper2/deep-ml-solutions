def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if len(np.unique(labels)) == 1:
        return True

    if depth >= max_depth:
        return True

    if len(labels) < min_samples_split:
        return True

    return False
