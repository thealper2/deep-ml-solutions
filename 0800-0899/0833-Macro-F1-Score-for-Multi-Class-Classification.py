import numpy as np

def macro_f1(y_true: list, y_pred: list) -> float:
    """
    Compute the Macro F1 score over all classes appearing in y_true or y_pred.
    Returns a single float.
    """
    classes = np.unique(np.concatenate((y_true, y_pred)))
    f1_scores = []

    for label in classes:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_pred != label) & (y_true == label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        f1_scores.append(f1)

    return np.mean(f1_scores)