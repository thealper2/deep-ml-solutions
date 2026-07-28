def add_bias_column(X):
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=X.dtype)
    return np.hstack([ones, X])
