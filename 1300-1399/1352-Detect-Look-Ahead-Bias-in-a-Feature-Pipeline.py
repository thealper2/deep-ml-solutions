import numpy as np


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score using only the trailing `window` values at each point.

    Args:
        x (np.ndarray): 1-D input series.
        window (int): lookback length, inclusive of the current point.

    Returns:
        np.ndarray: same length as x; the first window-1 entries are np.nan.
    """
    x = np.asarray(x)
    n = len(x)
    result = np.full(n, np.nan, dtype=float)

    for t in range(window - 1, n):
        start = max(0, t - window + 1)
        window_data = x[start:t+1]
        mu = np.mean(window_data)
        sigma = np.std(window_data, ddof=0)
        if sigma > 0:
            result[t] = (x[t] - mu) / sigma
        else:
            result[t] = 0.0
    
    return result


def has_lookahead(feature_fn, x: np.ndarray) -> bool:
    """Return True if feature_fn uses information from the future."""
    x = np.asarray(x)
    n = len(x)
    base = feature_fn(x)
    x_perturbed = x.copy()
    x_perturbed[-1] += 1000.0
    new = feature_fn(x_perturbed)
    return not np.allclose(base[:-1], new[:-1], equal_nan=True)