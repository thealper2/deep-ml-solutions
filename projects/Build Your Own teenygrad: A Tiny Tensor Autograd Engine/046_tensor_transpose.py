def tensor_transpose(x, ax1=-2, ax2=-1):
    ndim = len(x.shape)
    a1 = ax1 + ndim if ax1 < 0 else ax1
    a2 = ax2 + ndim if ax2 < 0 else ax2

    order = list(range(ndim))
    order[a1], order[a2] = order[a2], order[a1]

    return x.permute(tuple(order))
