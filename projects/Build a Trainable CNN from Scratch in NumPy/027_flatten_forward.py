def flatten_forward(x):
    N, C, H, W = x.shape
    out = x.reshape(N, -1)
    cache = {'x_shape': x.shape}
    return out, cache
