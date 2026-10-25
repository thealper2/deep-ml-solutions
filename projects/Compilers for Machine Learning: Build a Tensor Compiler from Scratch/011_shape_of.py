from functools import lru_cache


def buffer(name, shape, dtype=dtypes.float32):
    return UOp(Ops.BUFFER, dtype, (), (name, tuple(int(s) for s in shape)))


def strides_of(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


@lru_cache(maxsize=None)
def shape_of(u):
    op = u.op
    if op is Ops.BUFFER:
        return u.arg[1]
    if op is Ops.CONST:
        return ()
    if op in (Ops.RESHAPE, Ops.EXPAND):
        return tuple(u.arg)
    if op is Ops.PERMUTE:
        src_shape = shape_of(u.src[0])
        return tuple(src_shape[i] for i in u.arg)
    if op is Ops.FLIP:
        return shape_of(u.src[0])
    if op is Ops.PAD:
        src_shape = shape_of(u.src[0])
        return tuple(s + lo + hi for s, (lo, hi) in zip(src_shape, u.arg))
    if op is Ops.SHRINK:
        src_shape = shape_of(u.src[0])
        return tuple(e - b for s, (b, e) in zip(src_shape, u.arg))
    if op is Ops.REDUCE_AXIS:
        src_shape = list(shape_of(u.src[0]))
        _, axes = u.arg
        for ax in axes:
            src_shape[ax] = 1
        return tuple(src_shape)
    if op in ALU:
        if op is Ops.WHERE:
            return shape_of(u.src[1])
        return shape_of(u.src[0])
    raise NotImplementedError(f"shape_of for {op}")