import numpy as np

def linear_backward(d_out, x, w):
    dx = np.dot(d_out, w.T)
    dw = np.dot(x.T, d_out)
    db = np.sum(d_out, axis=0, keepdims=True)
    return dx, dw, db[0]
