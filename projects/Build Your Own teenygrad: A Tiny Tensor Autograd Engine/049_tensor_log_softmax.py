def tensor_log_softmax(x, axis=-1):
    ndim = len(x.shape)
    ax = axis + ndim if axis < 0 else axis

    m = Max.apply(x, axis=ax).expand(x.shape)
    shifted = Sub.apply(x, m)

    e = Exp.apply(shifted)
    s = Sum.apply(e, axis=ax)
    log_s = Log.apply(s).expand(x.shape)

    result = Sub.apply(shifted, log_s)
    result.lazydata._np = result.lazydata._np.astype(np.float64)
    return result
