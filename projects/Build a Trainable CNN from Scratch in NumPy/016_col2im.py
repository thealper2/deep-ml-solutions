def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape

    H_out = output_spatial_size(H, kernel_h, stride, padding)
    W_out = output_spatial_size(W, kernel_w, stride, padding)
    
    H_pad = H + 2 * padding
    W_pad = W + 2 * padding
    output_pad = np.zeros((N, C, H_pad, W_pad))

    patch_size = C * kernel_h * kernel_w

    patch_idx = 0
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                patch = cols[patch_idx].reshape(C, kernel_h, kernel_w)

                h_start = h_out * stride
                h_end = h_start + kernel_h
                w_start = w_out * stride
                w_end = w_start + kernel_w

                output_pad[n, :, h_start:h_end, w_start:w_end] += patch
                patch_idx += 1

    if padding > 0:
        return output_pad[:, :, padding:-padding, padding:-padding]

    return output_pad
