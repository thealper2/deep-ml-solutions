def conv2d_forward(x, weights, bias, stride, padding):
    N, C, H, W = x.shape
    C_out, C_in, kernel_h, kernel_w = weights.shape

    cols = im2col(x, kernel_h, kernel_w, stride, padding)

    weights_flat = weights.reshape(C_out, -1)

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    output_flat = cols @ weights_flat.T + bias.reshape(1, -1)

    output = output_flat.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

    cache = {
        'x_shape': x.shape,
        'weights': weights,
        'cols': cols,
        'stride': stride,
        'padding': padding,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
    }

    return output, cache
