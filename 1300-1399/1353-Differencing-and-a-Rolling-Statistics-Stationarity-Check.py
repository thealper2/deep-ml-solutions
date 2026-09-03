import numpy as np


def difference(x: np.ndarray, d: int) -> np.ndarray:
    """Apply first differencing d times.

    Args:
        x (np.ndarray): 1-D input series.
        d (int): number of differencing passes; 0 returns x unchanged.

    Returns:
        np.ndarray: series of length len(x) - d.
    """
    x = np.asarray(x)
    if d == 0:
        return x.copy()

    return np.diff(x, n=d)


def stationarity_report(x: np.ndarray, n_chunks: int) -> tuple:
    """Return (mean_spread, std_ratio) across n_chunks equal contiguous chunks,
    each rounded to 4 decimals."""
    x = np.asarray(x)
    n = len(x)

    chunk_size = n // n_chunks

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(x[start:end])

    means = [np.mean(chunk) for chunk in chunks]
    stds = [np.std(chunk, ddof=0) for chunk in chunks]

    mean_spread = max(means) - min(means)

    min_std = min(stds)
    max_std = max(stds)

    if min_std == 0:
        if max_std == 0:
            std_ratio = 1.0
        else:
            std_ratio = float('inf')
    else:
        std_ratio = max_std / min_std

    return round(float(mean_spread), 4), round(float(std_ratio), 4)