def linear_forward(x, w):
    y = x @ w
    return {'y': y, 'cache': {'x': x, 'w': w}}
