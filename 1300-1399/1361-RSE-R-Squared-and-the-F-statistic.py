import numpy as np


def model_fit_stats(X: np.ndarray, y: np.ndarray) -> tuple:
    """Residual standard error, R-squared and the overall F-statistic.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (rse, r2, f_stat) as floats. RSS == 0 gives (0.0, 1.0, inf).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    D = np.column_stack([np.ones(n), X])
    p = X.shape[1]

    coeffs, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
    y_pred = D @ coeffs

    residuals = y - y_pred
    RSS = np.sum(residuals ** 2)

    TSS = np.sum((y - np.mean(y)) ** 2)

    if TSS == 0:
        r2 = 1.0
    else:
        r2 = 1.0 - RSS / TSS

    if n - p - 1 > 0:
        rse = np.sqrt(RSS / (n - p - 1))
    else:
        rse = 0.0

    if RSS < 1e-12:
        f_stat = float('inf')
    else:
        f_stat = ((TSS - RSS) / p) / (RSS / (n - p - 1))

    return float(rse), float(r2), float(f_stat)