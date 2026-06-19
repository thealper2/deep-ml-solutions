def linear_forward(x, weights, bias):
    out = x @ weights + bias
    cache = {'x': x, 'weights': weights}
    return out, cache
