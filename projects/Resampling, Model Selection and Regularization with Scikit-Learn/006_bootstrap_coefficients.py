import numpy as np
from sklearn.linear_model import LinearRegression


def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = X_arr.shape[0]
    rng = np.random.default_rng(random_state)
    p = X_arr.shape[1]
    coefs = np.empty((n_boot, p), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        model = LinearRegression()
        model.fit(X_arr[idx], y_arr[idx])
        coefs[i] = model.coef_
    return coefs


def bootstrap_se(coefs):
    return np.round(np.std(np.asarray(coefs), axis=0, ddof=1), 2)


def ols_standard_errors(X, y):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = X_arr.shape[0]
    p = X_arr.shape[1]
    A = np.hstack([np.ones((n, 1)), X_arr])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    resid = y_arr - A @ coef
    rss = float(np.sum(resid ** 2))
    sigma2 = rss / (n - p - 1)
    ATA_inv = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(ATA_inv))
    return np.round(se[1:], 2)