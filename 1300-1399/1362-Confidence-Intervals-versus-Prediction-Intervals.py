import numpy as np


def interval_halfwidths(X: np.ndarray, y: np.ndarray, x0: np.ndarray, z: float) -> tuple:
    """Half-widths of the mean-response and new-observation intervals at x0.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.
        x0 (np.ndarray): (p,) point at which to form the intervals.
        z (float): multiplier, e.g. 1.96 for approximately 95%.

    Returns:
        tuple: (ci_half, pi_half) as floats.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    x0 = np.asarray(x0)
    n = len(y)
    p = X.shape[1]

    D = np.column_stack([np.ones(n), X])

    coeffs, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
    y_pred = D @ coeffs

    residuals = y - y_pred
    RSS = np.sum(residuals ** 2)
    sigma2 = RSS / (n - p - 1)

    d0 = np.concatenate([[1.0], x0])

    DtD = D.T @ D
    DtD_inv = np.linalg.inv(DtD)

    var_y0 = sigma2 * (d0 @ DtD_inv @ d0)
    ci_half = z * np.sqrt(var_y0)

    pi_half = z * np.sqrt(sigma2 + var_y0)

    return float(ci_half), float(pi_half)