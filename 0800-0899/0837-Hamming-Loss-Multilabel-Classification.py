import numpy as np

def hamming_loss(y_true, y_pred):
    """
    Compute the Hamming loss between two binary indicator matrices.
    Returns a single float.
    """
    if not y_true or not y_pred:
        return 0.0

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))