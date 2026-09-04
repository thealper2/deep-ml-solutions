import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    y_log_y_over_mu = np.where(
        y > 0,
        y * np.log(y / mu),
        0.0
    )
    deviance = 2.0 * np.sum(y_log_y_over_mu - (y - mu))
    return float(deviance)

def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    n = len(y)
    df = n - n_params
    pearson = np.sum((y - mu) ** 2 / mu)
    ratio = pearson / df
    return float(ratio)