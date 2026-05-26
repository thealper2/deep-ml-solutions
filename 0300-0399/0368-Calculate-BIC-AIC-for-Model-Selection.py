import numpy as np

def calculate_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple:
    """
    Calculate AIC and BIC for model selection.
    
    Args:
        y_true: True target values
        y_pred: Predicted values from the model
        k: Number of parameters in the model
    
    Returns:
        Tuple of (AIC, BIC)
    """
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    sigma_sq = rss / n
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return round(aic , 4), round(bic, 4)