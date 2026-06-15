def relu_backward(d_out, z):
    return d_out * (z > 0)
