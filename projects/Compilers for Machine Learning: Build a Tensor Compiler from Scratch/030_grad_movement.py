def grad_movement(u, g):
    op = u.op
    T = Tensor

    if op is Ops.RESHAPE:
        src = u.src[0]
        return [(src, g.reshape(*shape_of(src)))]

    if op is Ops.EXPAND:
        src = u.src[0]
        src_shape = shape_of(src)
        axes = [k for k, s in enumerate(src_shape) if s == 1 and shape_of(u)[k] != 1]
        if axes:
            g = g.sum(tuple(axes), keepdim=True)
        return [(src, g.reshape(*src_shape))]

    if op is Ops.PERMUTE:
        perm = u.arg
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return [(u.src[0], g.permute(*inv))]

    if op is Ops.FLIP:
        return [(u.src[0], g.flip(*u.arg))]

    if op is Ops.PAD:
        src = u.src[0]
        src_shape = shape_of(src)
        ranges = tuple((lo, lo + s) for (lo, hi), s in zip(u.arg, src_shape))
        return [(src, g.shrink(ranges))]

    if op is Ops.SHRINK:
        src = u.src[0]
        src_shape = shape_of(src)
        ranges = u.arg
        pads = tuple((b, s - e) for (b, e), s in zip(ranges, src_shape))
        return [(src, g.pad(pads))]

    if op is Ops.REDUCE_AXIS:
        src = u.src[0]
        src_shape = shape_of(src)
        red_op, axes = u.arg
        ge = g.expand(*src_shape)
        if red_op is Ops.ADD:
            return [(src, ge)]
        if red_op is Ops.MAX:
            src_t = T(src)
            max_t = T(u).expand(*src_shape)
            mask = (src_t < max_t).where(0.0, 1.0)
            return [(src, ge * mask)]
        raise NotImplementedError(f"reduce {red_op}")

    raise NotImplementedError(f"grad_movement for {op}")