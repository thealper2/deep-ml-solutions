def make_toy_digit_dataset(num_samples, seed=0):
    prototypes = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
    ], dtype=np.float32)

    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=num_samples)
    noise = rng.randn(num_samples, 9) * 0.1

    X = (prototypes[y] + noise).astype(np.float32)
    y = y.astype(np.int64)

    return X, y
