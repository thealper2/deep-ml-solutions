def make_shuffled_indices(n_samples, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return indices
