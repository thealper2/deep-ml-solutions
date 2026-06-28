def tensor_softmax(x, axis=-1):
    ndim = len(x.shape)
    ax = axis + ndim if axis < 0 else axis

    m = Max.apply(x, axis=ax).expand(x.shape)
    shifted = Sub.apply(x, m)

    e = Exp.apply(shifted)

    s = Sum.apply(e, axis=ax).expand(x.shape)
    return Div.apply(e, s)
