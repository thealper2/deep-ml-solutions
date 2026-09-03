import numpy as np


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation function.

    Args:
        x (np.ndarray): 1-D series.
        nlags (int): highest lag to compute.

    Returns:
        np.ndarray: length nlags + 1, entry 0 equal to 1.0.
    """
    x = np.asarray(x)
    n = len(x)
    x_mean = np.mean(x)

    denom = np.sum((x - x_mean) ** 2)

    if denom == 0:
        return np.array([[1.0] + [0.0] * nlags])

    acf_vals = np.zeros(nlags + 1)
    acf_vals[0] = 1.0

    for k in range(1, nlags + 1):
        num = np.sum((x[k:] - x_mean) * (x[:n-k] - x_mean))
        acf_vals[k] = num / denom

    return acf_vals

def ljung_box(x: np.ndarray, nlags: int) -> float:
    """Ljung-Box Q statistic over lags 1..nlags."""
    x = np.asarray(x)
    n = len(x)

    rho = acf(x, nlags)

    Q = 0.0
    for k in range(1, nlags + 1):
        Q += (rho[k] ** 2) / (n - k)

    Q = n * (n + 2) * Q

    return float(Q)