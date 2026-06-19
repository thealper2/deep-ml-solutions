def relu_backward(d_out, cache):
    x = cache['x']
    return d_out * (x > 0)
