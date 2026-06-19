def maxpool2d_backward(d_out, cache):
    x_shape = cache['x_shape']
    argmax = cache['argmax']
    kernel = cache['kernel']
    stride = cache['stride']

    N, C, H, W = x_shape
    _, _, out_h, out_w = d_out.shape

    dx = np.zeros(x_shape)

    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    grad_val = d_out[n, c, h, w]
                    argmax_idx = argmax[n, c, h, w]
                    window_grad = scatter_grad_window(grad_val, argmax_idx, kernel)
                    h_start = h * stride
                    w_start = w * stride
                    dx[n, c, h_start:h_start+kernel, w_start:w_start+kernel] += window_grad

    return dx
