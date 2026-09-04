import numpy as np


def fit_poisson_irls(X: np.ndarray, y: np.ndarray, n_iter: int) -> np.ndarray:
    """Fit log(lambda) = D @ beta by iteratively reweighted least squares.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) non-negative counts.
        n_iter (int): number of IRLS iterations, starting from beta = 0.

    Returns:
        np.ndarray: (p + 1,) coefficients, intercept first.
    """
    n = len(y)
    p = X.shape[1] if X.ndim > 1 else 0

    D = np.column_stack([np.ones(n), X])
    p_full = p + 1

    beta = np.zeros(p_full)

    for _ in range(n_iter):
        eta = D @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        w = mu
        W_D = w[:, None] * D
        DtWD = D.T @ W_D
        DtWz = D.T @ (w * z)
        beta = np.linalg.solve(DtWD, DtWz)

    return beta
