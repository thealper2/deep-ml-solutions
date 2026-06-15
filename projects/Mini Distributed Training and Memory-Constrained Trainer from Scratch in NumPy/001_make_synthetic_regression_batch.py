def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    np.random.seed(seed)
    x = np.random.randn(batch_size, in_dim).astype(np.float64)
    W_true = np.random.randn(in_dim, out_dim)
    noise = 0.1 * np.random.randn(batch_size, out_dim)
    y = x @ W_true + noise
    return x.astype(np.float64), y.astype(np.float64)
