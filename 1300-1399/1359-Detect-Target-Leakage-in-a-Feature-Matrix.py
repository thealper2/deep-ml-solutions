import numpy as np


def find_leaky_features(X: np.ndarray, y: np.ndarray, threshold: float) -> list:
    """Column indices whose |correlation| with y is >= threshold.

    Args:
        X (np.ndarray): (n, p) feature matrix.
        y (np.ndarray): (n,) target.
        threshold (float): absolute-correlation cutoff.

    Returns:
        list[int]: ascending column indices.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    p = X.shape[1]

    leaky_indices = []

    y_mean = np.mean(y)
    y_centered = y - y_mean
    y_var = np.sum(y_centered ** 2)

    for j in range(p):
        col = X[:, j]
        col_mean = np.mean(col)
        col_centered = col - col_mean

        col_var = np.sum(col_centered ** 2)
        if col_var < 1e-12:
            continue

        cov = np.sum(col_centered * y_centered)
        corr = cov / np.sqrt(col_var * y_var)

        if abs(corr) >= threshold:
            leaky_indices.append(j)

    return sorted(leaky_indices)

def is_affine_function_of(col: np.ndarray, y: np.ndarray) -> bool:
    """True when col == a + b * y to within a residual of 1e-9."""
    col = np.asarray(col)
    y = np.asarray(y)
    n = len(y)

    D = np.column_stack([np.ones(n), y])

    coeffs, _, _, _ = np.linalg.lstsq(D, col, rcond=None)
    col_pred = D @ coeffs

    residuals = col - col_pred
    RSS = np.sum(residuals ** 2)

    return RSS < 1e-9