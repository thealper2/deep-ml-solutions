import numpy as np


def baseline_metrics(y_train, y_test):
    """Accuracy and recall of always predicting the majority class of y_train."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    ones = int((y_train == 1).sum())
    zeros = int((y_train == 0).sum())
    majority = 1 if ones > zeros else 0

    preds = np.full(len(y_test), majority)
    accuracy = float((preds == y_test).mean())

    actual_pos = int((y_test == 1).sum())
    if actual_pos == 0:
        recall = 0.0
    else:
        recall = float(((preds == 1) & (y_test == 1)).sum() / actual_pos)

    base_rate = float(ones / len(y_train))

    return {
        "majority_class": int(majority),
        "accuracy": accuracy,
        "recall": recall,
        "base_rate": base_rate,
    }