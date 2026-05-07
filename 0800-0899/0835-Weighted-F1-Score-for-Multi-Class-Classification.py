import numpy as np

def weighted_f1_score(y_true, y_pred) -> float:
    """
    Compute the weighted F1 score for multi-class classification.

    Args:
        y_true: array-like of true class labels
        y_pred: array-like of predicted class labels

    Returns:
        Weighted F1 score rounded to 4 decimal places.
    """
    classes = np.unique(y_true)
    f1_list = []
    support_list = []

    for c in classes:
        true_c = (y_true == c)
        pred_c = (y_pred == c)
        tp = np.sum(true_c & pred_c)
        fp = np.sum(~true_c & pred_c)
        fn = np.sum(true_c & ~pred_c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        f1_list.append(f1)
        support_list.append(np.sum(true_c))

    weighted_f1 = np.average(f1_list, weights=support_list)
    return weighted_f1