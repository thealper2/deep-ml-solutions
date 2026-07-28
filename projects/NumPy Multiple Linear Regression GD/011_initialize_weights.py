def initialize_weights(n_features, seed=None):
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(0, 0.01, n_features)
