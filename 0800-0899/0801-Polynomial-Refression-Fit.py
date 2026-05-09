import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    X = np.zeros((n, degree + 1))
    for i in range(degree + 1):
        X[:, i] = x ** i

    coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
    return coeffs.tolist()
