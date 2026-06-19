def conv2d_grad_bias(d_out):
    return np.sum(d_out, axis=(0, 2, 3))
