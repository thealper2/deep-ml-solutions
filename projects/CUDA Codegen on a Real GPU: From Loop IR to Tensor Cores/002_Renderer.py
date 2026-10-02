import math


def c_lit(dtype, v):
    if dtype in (dtypes.int32, dtypes.bool): return str(int(v))
    v = float(v)
    if math.isinf(v): return "INFINITY" if v > 0 else "(-INFINITY)"
    return repr(v) + "f"


class Renderer:
    _infix = {Ops.ADD: "+", Ops.MUL: "*", Ops.CMPLT: "<", Ops.AND: "&&", Ops.IDIV: "/", Ops.MOD: "%"}

    def __init__(self): self.lines, self.scopes, self.n = [], [{}], 0

    def push(self): self.scopes.append({})
    def pop(self): self.scopes.pop()
    def emit(self, s): self.lines.append("  " * len(self.scopes) + s)

    def var(self, u, e):
        name = f"v{self.n}"; self.n += 1
        self.emit(f"{u.dtype.c} {name} = {e};")
        self.scopes[-1][u] = name
        return name

    def expr(self, u):
        for sc in reversed(self.scopes):
            if u in sc: return sc[u]
        op = u.op
        if op is Ops.CONST: return c_lit(u.dtype, u.arg)
        if op is Ops.RANGE: return f"r{u.arg}"
        if op is Ops.SPECIAL: return u.arg
        if op is Ops.DEFINE_ACC: return f"acc{u.arg}"
        if op is Ops.LOAD:
            p, idx = u.src[0].src
            buf = f"data{p.arg[1]}" if p.op is Ops.PARAM else p.arg[0]
            return self.var(u, f"{buf}[{self.expr(idx)}]")
        if op is Ops.WHERE:
            c, a, b = (self.expr(s) for s in u.src)
            return self.var(u, f"({c} ? {a} : {b})")
        if op is Ops.RECIP: return self.var(u, f"(1.0f/{self.expr(u.src[0])})")
        if op is Ops.EXP2: return self.var(u, f"exp2f({self.expr(u.src[0])})")
        if op is Ops.SQRT: return self.var(u, f"sqrtf({self.expr(u.src[0])})")
        if op is Ops.MAX:
            a, b = (self.expr(s) for s in u.src)
            fn = "max" if u.dtype in (dtypes.int32, dtypes.bool) else "fmaxf"
            return self.var(u, f"{fn}({a}, {b})")
        if op in self._infix:
            a, b = (self.expr(s) for s in u.src)
            return self.var(u, f"({a} {self._infix[op]} {b})")
        raise NotImplementedError(op)