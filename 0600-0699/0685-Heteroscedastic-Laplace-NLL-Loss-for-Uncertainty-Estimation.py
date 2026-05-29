import numpy as np

def laplace_nll_loss(y_true, mu, log_b):
    """
    Compute the mean heteroscedastic Laplace negative log-likelihood loss.
    
    Args:
        y_true: 1D numpy array of true target values, shape (n,)
        mu: 1D numpy array of predicted locations, shape (n,)
        log_b: 1D numpy array of predicted log-scale parameters, shape (n,)
    
    Returns:
        float: mean NLL loss over the batch
    """
    b = np.exp(log_b)
    abs_error = np.abs(y_true - mu)
    all_components = (abs_error / b) + log_b + np.log(2.0)
    nll = np.mean(all_components)
    return round(nll, 4)