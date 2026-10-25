def special(n, i, name):
    return UOp(Ops.SPECIAL, dtypes.int32, (UOp.const(dtypes.int32, n),), (i, name))


def to_gpu(k):
    assert 1 <= len(k.out_ranges) <= 3, "to_gpu requires 1..3 output ranges"
    n = len(k.out_ranges)
    names = ["gidx0", "gidx1", "gidx2"]  # last axis -> gidx0 (x), then gidx1, gidx2
    specials = [None] * n
    for pos in range(n):
        # pos 0 is the outermost output range (z if 3, y if 2, x if 1)
        # x is the innermost (last) output range
        axis_from_inner = n - 1 - pos
        specials[pos] = special(int(k.out_ranges[pos].src[0].arg), k.out_ranges[pos].arg, names[axis_from_inner])
    mapping = {k.out_ranges[pos]: specials[pos] for pos in range(n)}
    k2 = k.map_exprs(lambda e: substitute(e, mapping))
    k2.out_ranges = specials
    return k2


def launch_dims(k, block=(32, 4, 1)):
    n = len(k.out_ranges)
    # extents by axis from innermost: x = last, y = before, z = before that
    exts = [0, 0, 0]  # x, y, z
    for pos, s in enumerate(k.out_ranges):
        axis_from_inner = n - 1 - pos
        exts[axis_from_inner] = int(s.src[0].arg)
    bx = min(block[0], exts[0]) if exts[0] > 0 else 1
    by = min(block[1], exts[1]) if exts[1] > 0 else 1
    bz = min(block[2], exts[2]) if exts[2] > 0 else 1
    gx = (exts[0] + bx - 1) // bx if exts[0] > 0 else 1
    gy = (exts[1] + by - 1) // by if exts[1] > 0 else 1
    gz = (exts[2] + bz - 1) // bz if exts[2] > 0 else 1
    return (gx, gy, gz), (bx, by, bz)