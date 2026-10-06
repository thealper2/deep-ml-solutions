from functools import lru_cache


@lru_cache(maxsize=None)
def bounds(u):
    op = u.op
    if op is Ops.CONST:
        if u.dtype is dtypes.float32:
            return (-INF, INF)
        v = u.arg
        if u.dtype is dtypes.bool:
            v = 1 if v else 0
        return (v, v)
    if op in (Ops.RANGE, Ops.SPECIAL):
        n = u.src[0].arg
        return (0, n - 1)
    if op is Ops.ADD:
        a, b = u.src
        amin, amax = bounds(a)
        bmin, bmax = bounds(b)
        return (amin + bmin, amax + bmax)
    if op is Ops.MUL:
        a, b = u.src
        amin, amax = bounds(a)
        bmin, bmax = bounds(b)
        if any(x in (INF, -INF) for x in (amin, amax, bmin, bmax)):
            if amax == amin == 0 or bmax == bmin == 0:
                return (0, 0)
            return (-INF, INF)
        products = [x * y for x in (amin, amax) for y in (bmin, bmax)]
        return (min(products), max(products))
    if op is Ops.MAX:
        a, b = u.src
        amin, amax = bounds(a)
        bmin, bmax = bounds(b)
        return (max(amin, bmin), max(amax, bmax))
    if op is Ops.IDIV:
        a, b = u.src
        amin, amax = bounds(a)
        bmin, bmax = bounds(b)
        if bmin == bmax and bmin > 0:
            c = bmin
            return (_floor_div(amin, c), _floor_div(amax, c))
        return (-INF, INF)
    if op is Ops.MOD:
        a, b = u.src
        amin, amax = bounds(a)
        bmin, bmax = bounds(b)
        if bmin == bmax and bmin > 0:
            c = bmin
            if amin >= 0 and amax < c:
                return (amin, amax)
            return (0, c - 1)
        return (-INF, INF)
    if op is Ops.CMPLT:
        a, b = u.src
        amin, amax = bounds(a)
        bmin, bmax = bounds(b)
        if amax < bmin:
            return (1, 1)
        if amin >= bmax:
            return (0, 0)
        return (0, 1)
    if op is Ops.AND:
        a, b = u.src
        amin, amax = bounds(a)
        bmin, bmax = bounds(b)
        if amin >= 1 and bmin >= 1:
            return (1, 1)
        if amax == 0 or bmax == 0:
            return (0, 0)
        return (0, 1)
    if op is Ops.WHERE:
        cond, a, b = u.src
        cmin, cmax = bounds(cond)
        if cmin == cmax:
            return bounds(a) if cmin >= 1 else bounds(b)
        alo, ahi = bounds(a)
        blo, bhi = bounds(b)
        return (min(alo, blo), max(ahi, bhi))
    if op is Ops.CAST:
        if u.dtype is dtypes.float32:
            return (-INF, INF)
        return bounds(u.src[0])
    return (-INF, INF)


def _floor_div(x, c):
    if x == INF or x == -INF:
        return x
    return x // c