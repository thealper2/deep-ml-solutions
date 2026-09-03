import numpy as np


def variance_inflation_factors(X: np.ndarray) -> np.ndarray:
    """VIF for each column of X, from an auxiliary regression on the other columns.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.

    Returns:
        np.ndarray: (p,) variance inflation factors.
    """
    X = np.asarray(X)
    n, p = X.shape

    if p == 1:
        return np.array([1.0])

    vifs = np.zeros(p)
    for j in range(p):
        other_cols = np.delete(X, j, axis=1)
        D = np.column_stack([np.ones(n), other_cols])
        y_j = X[:, j]
        coeffs, _, _, _ = np.linalg.lstsq(D, y_j, rcond=None)
        y_pred = D @ coeffs
        residuals = y_j - y_pred
        RSS = np.sum(residuals ** 2)
        TSS = np.sum((y_j - np.mean(y_j)) ** 2)

        if TSS < 1e-12:
            vifs[j] = float('inf')
            continue

        r2 = 1.0 - RSS / TSS

        if r2 > 1.0 - 1e-12:
            vifs[j] = float('inf')
        else:
            vifs[j] = 1.0 / (1.0 - r2)

    return vifs