def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    x = cache['x']
    dW = np.dot(x.T, dy)
    return dW
