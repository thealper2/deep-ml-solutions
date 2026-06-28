def tensor_mean(x, axis=None, keepdim=False):
    ndim = len(x.shape)

    if axis is None:
        axes = list(range(ndim))
        norm_axis = None
    elif isinstance(axis, int):
        a = axis + ndim if axis < 0 else axis
        axes = [a]
        norm_axis = a
    else:
        axes = [(a + ndim if a < 0 else a) for a in axis]
        norm_axis = tuple(axes)

    count = 1
    for a in axes:
        count *= x.shape[a]

    summed = Sum.apply(x, axis=norm_axis)

    divisor = Tensor(LazyBuffer.const(float(count), summed.shape), requires_grad=False)
    result = Div.apply(summed, divisor)

    if not keepdim:
        new_shape = list(result.shape)
        for a in sorted(axes, reverse=True):
            del new_shape[a]
        result = result.reshape(tuple(new_shape))

    return result
