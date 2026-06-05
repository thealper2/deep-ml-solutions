import numpy as np

def distance_correlation_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute squared distance correlation between X and Y.
    
    Args:
        X: Array of shape (n_samples,) or (n_samples, n_features)
        Y: Array of shape (n_samples,) or (n_samples, n_features)
    
    Returns:
        dcor^2 value between 0 and 1
    """
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    n = X.shape[0]
    X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D_X = np.sqrt(np.sum(X_diff ** 2, axis=2))
    Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    D_Y = np.sqrt(np.sum(Y_diff ** 2, axis=2))
    row_mean_X = np.mean(D_X, axis=1, keepdims=True)
    col_mean_X = np.mean(D_X, axis=0, keepdims=True)
    grand_mean_X = np.mean(D_X)
    A = D_X - row_mean_X - col_mean_X + grand_mean_X
    row_mean_Y = np.mean(D_Y, axis=1, keepdims=True)
    col_mean_Y = np.mean(D_Y, axis=0, keepdims=True)
    grand_mean_Y = np.mean(D_Y)
    B = D_Y - row_mean_Y - col_mean_Y + grand_mean_Y
    dCov_sq = np.sum(A * B) / (n ** 2)
    dVar_X_sq = np.sum(A * A) / (n ** 2)
    dVar_Y_sq = np.sum(B * B) / (n ** 2)

    if dVar_X_sq == 0 or dVar_Y_sq == 0:
        dCor_sq = 0.0
    else:
        dCor_sq = dCov_sq / np.sqrt(dVar_X_sq * dVar_Y_sq)

    return round(dCor_sq, 4)
    