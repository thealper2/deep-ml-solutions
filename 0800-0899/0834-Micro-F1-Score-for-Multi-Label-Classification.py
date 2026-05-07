import numpy as np

def micro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the micro-averaged F1 score for multi-label classification.

    Args:
        y_true: Binary indicator array of shape (n_samples, n_labels)
        y_pred: Binary indicator array of shape (n_samples, n_labels)

    Returns:
        Micro-averaged F1 score as a float.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1