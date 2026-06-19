import numpy as np

def linear_grad_bias(dout):
    return np.sum(dout, axis=0)
