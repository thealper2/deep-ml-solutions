import numpy as np

def permutation_importance(X, y, weights, bias, permutations):
    """
    Compute permutation feature importance for a linear regression model.

    Args:
        X: 2D array-like of shape (n_samples, n_features)
        y: 1D array-like of shape (n_samples,)
        weights: 1D array-like of shape (n_features,)
        bias: float, the intercept
        permutations: list of length n_features; permutations[j] is a list of
                      permutation index arrays to apply to column j

    Returns:
        List of feature importances (length n_features).
    """
    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    n_samples = len(y)

    y_pred_baseline = X @ weights + bias
    ss_res_baseline = np.sum((y - y_pred_baseline) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    baseline_r2 = 1 - (ss_res_baseline / ss_tot) if ss_tot > 0 else 0.0

    n_features = X.shape[1]
    importances = []

    for j in range(n_features):
        perm_r2s = []
        for perm in permutations[j]:
            X_perm = X.copy()
            X_perm[:, j] = X_perm[perm, j]
            y_pred_perm = X_perm @ weights + bias
            ss_res_perm = np.sum((y - y_pred_perm) ** 2)
            r2_perm = 1 - (ss_res_perm / ss_tot) if ss_tot > 0 else 0.0
            perm_r2s.append(r2_perm)

        mean_perm_r2 = np.mean(perm_r2s)
        importance = baseline_r2 - mean_perm_r2
        importances.append(importance)

    return importances