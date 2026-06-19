def conv2d_grad_weights(d_out, cache):
    cols = cache['cols']
    weights_shape = cache['weights'].shape
    C_out, C_in, kH, kW = weights_shape

    N, C_out, out_h, out_w = d_out.shape
    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    dW_flat = cols.T @ d_out_flat

    dW = dW_flat.T.reshape(C_out, C_in, kH, kW)
    return dW
