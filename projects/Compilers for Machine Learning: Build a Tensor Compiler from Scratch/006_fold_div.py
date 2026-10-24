def _terms(u):
    if u.op is Ops.ADD:
        return _terms(u.src[0]) + _terms(u.src[1])
    return [u]


def _factor(t):
    if t.op is Ops.CONST:
        return (None, t.arg)
    if t.op is Ops.MUL:
        a, b = t.src
        if a.op is Ops.CONST:
            return (b, a.arg)
        if b.op is Ops.CONST:
            return (a, b.arg)
    return (t, 1)


def _sum(terms, dtype):
    if not terms:
        return UOp.const(dtype, 0)
    result = terms[0]
    for t in terms[1:]:
        result = result + t
    return result


def _rebuild(factors, dtype):
    terms = []
    for base, k in factors:
        if base is None:
            terms.append(UOp.const(dtype, k))
        elif k == 1:
            terms.append(base)
        else:
            terms.append(base * k)
    return _sum(terms, dtype)


def fold_div(x, c):
    c = c.arg if c.op is Ops.CONST else None
    if c is None or c <= 0:
        return None
    terms = _terms(x)
    divisible = []
    rest = []
    for t in terms:
        base, k = _factor(t)
        if k % c == 0:
            divisible.append((base, k // c))
        else:
            rest.append(t)
    R = _sum(rest, x.dtype)
    rmin, rmax = bounds(R)
    if not divisible:
        if len(terms) == 1:
            base, k = _factor(terms[0])
            if base is not None and k > 1 and c % k == 0:
                return base // (c // k)
        if 0 <= rmin and rmax < c:
            return UOp.const(x.dtype, 0)
        return None
    quot = _rebuild(divisible, x.dtype)
    if 0 <= rmin and rmax < c:
        return quot
    if R.op is Ops.CONST:
        q = UOp.const(x.dtype, R.arg // c)
    else:
        q = R // UOp.const(x.dtype, c)
    if quot.op is Ops.CONST and quot.arg == 0:
        return q
    if q.op is Ops.CONST and q.arg == 0:
        return quot
    return quot + q


def fold_mod(x, c):
    c = c.arg if c.op is Ops.CONST else None
    if c is None or c <= 0:
        return None
    terms = _terms(x)
    kept = []
    dropped = False
    for t in terms:
        base, k = _factor(t)
        if k % c == 0:
            dropped = True
        else:
            kept.append(t)
    R = _sum(kept, x.dtype)
    rmin, rmax = bounds(R)
    if 0 <= rmin and rmax < c:
        return R
    if dropped:
        return R % UOp.const(x.dtype, c)
    return None


def fold_cmplt(x, y):
    lo, hi = bounds(x < y)
    if lo == hi:
        return UOp.const(dtypes.bool, bool(lo))
    return None


def fold_max(x, y):
    xlo, xhi = bounds(x)
    ylo, yhi = bounds(y)
    if xlo >= yhi:
        return x
    if ylo >= xhi:
        return y
    return None