import numpy as np

def flatten_backward(d_out, cache):
    x_shape = cache['x_shape']
    return d_out.reshape(x_shape)
