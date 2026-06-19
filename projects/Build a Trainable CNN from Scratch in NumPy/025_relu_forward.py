def relu_forward(x):
    y = np.maximum(0, x)
    cache = {'x': x}
    return y, cache
