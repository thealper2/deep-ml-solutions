import numpy as np

def make_windows(series, window, horizon, stride=1):
    """
    Returns (X, Y): X (n_windows, window), Y (n_windows, horizon).
    Empty arrays with shapes (0, window) and (0, horizon) if the series is too short.
    """
    series = np.asarray(series)
    T = len(series)

    if T < window + horizon:
        X = np.empty((0, window), dtype=series.dtype)
        Y = np.empty((0, horizon), dtype=series.dtype)
        return X, Y

    n_windows = (T - window - horizon) // stride + 1

    if n_windows <= 0:
        X = np.empty((0, window), dtype=series.dtype)
        Y = np.empty((0, horizon), dtype=series.dtype)
        return X, Y

    X = np.zeros((n_windows, window), dtype=series.dtype)
    Y = np.zeros((n_windows, horizon), dtype=series.dtype)

    for i in range(n_windows):
        start = i * stride
        X[i] = series[start:start + window]
        Y[i] = series[start + window:start + window + horizon]

    return X, Y