import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Iterable of true labels in [0, K-1]
        y_pred: Iterable of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        list[list[int|float]] confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.float64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    if normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm = cm / col_sums
    elif normalize == 'all':
        cm = cm / cm.sum()

    return cm