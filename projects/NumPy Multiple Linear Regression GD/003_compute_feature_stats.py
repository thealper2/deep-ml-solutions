def compute_feature_stats(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std
