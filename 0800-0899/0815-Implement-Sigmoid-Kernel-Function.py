import numpy as np

def sigmoid_kernel(X1, X2, alpha=0.01, c=0.0):
    """
    Compute the sigmoid kernel matrix between rows of X1 and rows of X2.

    Args:
        X1: numpy array of shape (n, d)
        X2: numpy array of shape (m, d)
        alpha: float, slope parameter
        c: float, intercept (constant) parameter

    Returns:
        Nested list of shape (n, m) with entries rounded to 4 decimals.
    """
    dot_product = np.dot(X1, X2.T)
    return np.tanh(alpha * dot_product + c)
