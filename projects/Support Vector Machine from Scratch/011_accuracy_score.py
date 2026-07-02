import numpy as np

def accuracy_score(y_pred, y_true):
    return (y_pred == y_true).sum() / len(y_true)
