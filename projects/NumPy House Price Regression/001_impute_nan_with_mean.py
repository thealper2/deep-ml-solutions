def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X_imputed = X.copy()
    for col in range(X.shape[1]):
        col_data = X[:, col]
        mask = ~np.isnan(col_data)
        if np.any(mask):
            mean_val = np.mean(col_data[mask])
            X_imputed[mask == False, col] = mean_val
        else:
            X_imputed[:, col] = 0.0

    return X_imputed
