def conv2d_grad_input(d_out, cache):
    x_shape = cache['x_shape']
    weights = cache['weights']
    stride = cache['stride']
    padding = cache['padding']
    kernel_h = cache['kernel_h']
    kernel_w = cache['kernel_w']

    N, C_in, H, W = x_shape
    C_out, C_in_k, kH, kW = weights.shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)

    weights_flat = weights.reshape(C_out, -1)
    d_cols = d_out_flat @ weights_flat

    dx = col2im(d_cols, x_shape, kernel_h, kernel_w, stride, padding)
    return dx
