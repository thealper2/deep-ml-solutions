import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; returns 0.0 when the variance of y_true is 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred
    RSS = np.sum(residuals ** 2)

    y_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_mean) ** 2)

    if TSS == 0:
        return 0.0

    return 1.0 - RSS / TSS

def rolling_r2(y_true: np.ndarray, y_pred: np.ndarray, window: int) -> np.ndarray:
    """r2_score over each contiguous window.

    Returns:
        np.ndarray: length len(y_true) - window + 1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    result = np.zeros(n - window + 1)

    for i in range(n - window + 1):
        y_true_window = y_true[i:i+window]
        y_pred_window = y_pred[i:i+window]
        result[i] = r2_score(y_true_window, y_pred_window)

    return result