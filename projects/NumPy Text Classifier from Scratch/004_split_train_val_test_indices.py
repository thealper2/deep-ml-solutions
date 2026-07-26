def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_val = int(n_samples * val_fraction)
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_val - n_test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:n_train+n_val+n_test]
    return train_idx, val_idx, test_idx
