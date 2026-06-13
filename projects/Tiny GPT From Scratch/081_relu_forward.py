def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    y = np.maximum(0, x)
    return {'y': y, 'cache': {'x': x}}
