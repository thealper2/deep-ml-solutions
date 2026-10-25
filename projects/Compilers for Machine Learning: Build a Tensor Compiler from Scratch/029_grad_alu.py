import math


def grad_alu(u, g):
    op = u.op
    T = Tensor

    if op is Ops.ADD:
        a, b = u.src
        return [(a, g), (b, g)]

    if op is Ops.MUL:
        a, b = u.src
        return [(a, g * T(b)), (b, g * T(a))]

    if op is Ops.MAX:
        a, b = u.src
        ta, tb = T(a), T(b)
        mask = tb < ta
        return [(a, mask.where(g, 0.0)), (b, mask.where(0.0, g))]

    if op is Ops.RECIP:
        (x,) = u.src
        tu = T(u)
        return [(x, -g * tu * tu)]

    if op is Ops.EXP2:
        (x,) = u.src
        tu = T(u)
        return [(x, g * tu * math.log(2))]

    if op is Ops.LOG2:
        (x,) = u.src
        tx = T(x)
        return [(x, g / (tx * math.log(2)))]

    if op is Ops.SQRT:
        (x,) = u.src
        tu = T(u)
        return [(x, g / (2.0 * tu))]

    if op is Ops.WHERE:
        c, a, b = u.src
        tc = T(c)
        return [(a, tc.where(g, 0.0)), (b, tc.where(0.0, g))]

    if op is Ops.CAST:
        (x,) = u.src
        return [(x, g.cast(x.dtype))]

    if op in (Ops.CMPLT, Ops.AND):
        return []

    raise NotImplementedError(f"grad_alu for {op}")