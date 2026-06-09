import numpy as np

def matthews_correlation_coefficient(y_true: list, y_pred: list) -> float:
    """
    Calculate the Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: List of actual binary labels (0 or 1)
        y_pred: List of predicted binary labels (0 or 1)
    
    Returns:
        MCC value rounded to 4 decimal places
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0

    mcc = numerator / denominator
    return round(mcc, 4)
