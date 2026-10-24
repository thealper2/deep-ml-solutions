import math


def exec_alu(op, dtype, vals):
    if op is Ops.ADD:
        r = vals[0] + vals[1]
    elif op is Ops.MUL:
        r = vals[0] * vals[1]
    elif op is Ops.MAX:
        r = max(vals[0], vals[1])
    elif op is Ops.CMPLT:
        r = vals[0] < vals[1]
    elif op is Ops.AND:
        r = bool(vals[0]) and bool(vals[1])
    elif op is Ops.IDIV:
        a, b = vals
        r = 0 if b == 0 else a // b
    elif op is Ops.MOD:
        a, b = vals
        r = 0 if b == 0 else a % b
    elif op is Ops.RECIP:
        x = vals[0]
        r = float('inf') if x == 0 else 1.0 / x
    elif op is Ops.EXP2:
        x = vals[0]
        r = INF if x >= 128 else 2.0 ** x
    elif op is Ops.LOG2:
        x = vals[0]
        if x == 0:
            r = -INF
        elif x < 0:
            r = float('nan')
        else:
            r = math.log2(x)
    elif op is Ops.SQRT:
        x = vals[0]
        r = float('nan') if x < 0 else math.sqrt(x)
    elif op is Ops.CAST:
        r = vals[0]
    elif op is Ops.WHERE:
        r = vals[1] if vals[0] else vals[2]
    else:
        r = vals[0] if vals else 0
    return UOp.const(dtype, r)


def basic_rules():
    rules = []

    def fn_fold(u):
        if u.op in ALU and all(s.op is Ops.CONST for s in u.src):
            return exec_alu(u.op, u.dtype, [s.arg for s in u.src])
        return None
    rules.append((UPat(ALU, name="u"), fn_fold))

    def fn_where(u):
        c, a, b = u.src
        if c.op is Ops.CONST:
            return a if c.arg else b
        if a is b:
            return a
        return None
    rules.append((UPat(Ops.WHERE, name="u"), fn_where))

    def make_move(op):
        def fn(a, b):
            if a.op is Ops.CONST and b.op is not Ops.CONST:
                return UOp(op, b.dtype, (b, a))
            return None
        return fn
    for op in (Ops.ADD, Ops.MUL, Ops.MAX, Ops.AND):
        rules.append((UPat(op, src=(UPat.cvar("a"), UPat.var("b"))), make_move(op)))

    rules.append((UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c"))),
                  lambda x, c: x if c.arg == 0 else None))
    rules.append((UPat(Ops.MUL, src=(UPat.var("x"), UPat.cvar("c"))),
                  lambda x, c: x if c.arg == 1 else (UOp.const(x.dtype, 0) if c.arg == 0 else None)))
    def fn_div_mod(u):
        x, c = u.src
        if c.op is Ops.CONST and c.arg == 1:
            return x if u.op is Ops.IDIV else UOp.const(x.dtype, 0)
        return None
    rules.append((UPat({Ops.IDIV, Ops.MOD}, src=(UPat.var(), UPat.cvar()), name="u"), fn_div_mod))
    rules.append((UPat(Ops.AND, src=(UPat.var("x"), UPat.cvar("c"))),
                  lambda x, c: x if c.arg else UOp.const(x.dtype, 0)))
    rules.append((UPat(Ops.MAX, src=(UPat.var("x"), UPat.var("x"))),
                  lambda x: x))

    rules.append((UPat(Ops.ADD, src=(UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c1"))), UPat.cvar("c2"))),
                  lambda x, c1, c2: UOp(Ops.ADD, x.dtype, (x, UOp.const(x.dtype, c1.arg + c2.arg)))))
    rules.append((UPat(Ops.MUL, src=(UPat(Ops.MUL, src=(UPat.var("x"), UPat.cvar("c1"))), UPat.cvar("c2"))),
                  lambda x, c1, c2: UOp(Ops.MUL, x.dtype, (x, UOp.const(x.dtype, c1.arg * c2.arg)))))
    rules.append((UPat(Ops.ADD, src=(UPat(Ops.ADD, src=(UPat.var("x"), UPat.cvar("c"))), UPat.var("y"))),
                  lambda x, c, y: (UOp(Ops.ADD, x.dtype, (x, y)) + c) if y.op is not Ops.CONST else None))
    rules.append((UPat(Ops.ADD, src=(UPat.var("x"), UPat(Ops.ADD, src=(UPat.var("y"), UPat.cvar("c"))))),
                  lambda x, y, c: (UOp(Ops.ADD, x.dtype, (x, y)) + c) if y.op is not Ops.CONST else None))

    rules.append((UPat(Ops.MUL, dtype=dtypes.int32,
                       src=(UPat(Ops.ADD, dtype=dtypes.int32, src=(UPat.var("a"), UPat.var("b"))), UPat.cvar("c"))),
                  lambda a, b, c: UOp(Ops.ADD, dtypes.int32, (a * c, b * c))))

    return PatternMatcher(rules)