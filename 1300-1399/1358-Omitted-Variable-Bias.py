import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    X_design = np.column_stack([np.ones(len(X)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    return coeffs


def omitted_variable_bias(X: np.ndarray, y: np.ndarray, omit_idx: int) -> tuple:
    """Return (full_kept, short, bias) for a two-column X."""
    X = np.asarray(X)
    y = np.asarray(y)
    full_coeffs = ols(X, y)
    kept_idx = 1 - omit_idx
    full_kept = full_coeffs[kept_idx + 1]
    X_kept = X[:, kept_idx:kept_idx+1]
    short_coeffs = ols(X_kept, y)
    short = short_coeffs[1]
    omitted_idx = omit_idx
    X_omitted = X[:, omitted_idx:omitted_idx+1]
    aux_coeffs = ols(X_kept, X_omitted.flatten())
    delta = aux_coeffs[1]
    beta_omitted = full_coeffs[omitted_idx+1]
    bias = beta_omitted * delta
    return full_kept, short, bias