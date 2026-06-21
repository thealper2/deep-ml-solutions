def iterate_minibatches(buffer, batch_size, seed=None):
    """Yield shuffled minibatches of step dicts of size <= batch_size."""
    n = len(buffer)
    
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    indices = rng.permutation(n).tolist()

    for i in range(0, n, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield [buffer[idx] for idx in batch_indices]
