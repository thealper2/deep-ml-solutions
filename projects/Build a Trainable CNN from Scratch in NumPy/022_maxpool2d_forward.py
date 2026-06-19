def maxpool2d_forward(x, kernel, stride):
    N, C, H, W = x.shape

    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.zeros((N, C, out_h, out_w))
    argmax = np.zeros((N, C, out_h, out_w), dtype=int)

    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    h_end = h_start + kernel
                    w_start = w * stride
                    w_end = w_start + kernel

                    window = x[n, c, h_start:h_end, w_start:w_end]

                    max_val = np.max(window)
                    flat_idx = np.argmax(window)

                    out[n, c, h, w] = max_val
                    argmax[n, c, h, w] = flat_idx

    cache = {
        'x_shape': x.shape,
        'argmax': argmax,
        'kernel': kernel,
        'stride': stride,
    }

    return out, cache
