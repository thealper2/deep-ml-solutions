def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X_imputed = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_imputed, k=iqr_k)
    X_clipped = clip_columns(X_imputed, lower, upper)
    return X_clipped
