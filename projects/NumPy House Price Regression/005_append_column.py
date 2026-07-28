def append_column(X, col):
    if col.ndim == 1:
        col = col.reshape(-1, 1)

    return np.hstack([X, col])
