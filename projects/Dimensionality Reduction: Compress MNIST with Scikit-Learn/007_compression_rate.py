def compression_ratio(X, X_reduced):
    return round(X.shape[1] / X_reduced.shape[1], 2)
