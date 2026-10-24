import numpy as np
import math


def c_literal(dtype, v):
    if dtype is dtypes.float32:
        f = float(np.float32(v))
        if math.isinf(f):
            return "INFINITY" if f > 0 else "(-INFINITY)"
        if math.isnan(f):
            return "NAN"
        return f"{f!r}f"
    if dtype is dtypes.bool:
        return "1" if v else "0"
    return str(int(v))


C_HEADER = """#include <math.h>

static inline int fdiv(int a, int b) {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

static inline int fmod_floor(int a, int b) {
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
"""


def param(name, dtype, i):
    return UOp(Ops.PARAM, dtype, (), (name, i))


def load(p, idx):
    return UOp(Ops.LOAD, p.dtype, (UOp(Ops.INDEX, dtypes.int32, (p, UOp.const(dtypes.int32, idx))),))


def store(p, idx, val):
    return UOp(Ops.STORE, None, (UOp(Ops.INDEX, dtypes.int32, (p, UOp.const(dtypes.int32, idx))), val))


def sink(*stores):
    return UOp(Ops.SINK, None, stores)


def params_of(u):
    seen = {}
    stack = [u]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op is Ops.PARAM:
            seen[node.arg[1]] = node
        for s in node.src:
            stack.append(s)
    return [seen[i] for i in sorted(seen)]


class CRenderer:
    def __init__(self):
        self.lines = []
        self.scopes = [{}]
        self.counter = 0

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def emit(self, line):
        indent = "    " * (len(self.scopes) - 1)
        self.lines.append(indent + line)

    def lookup(self, u):
        for scope in reversed(self.scopes):
            if u in scope:
                return scope[u]
        return None

    def new_var(self, u, expr, dtype=None):
        dt = dtype if dtype is not None else u.dtype
        name = f"v{self.counter}"
        self.counter += 1
        self.emit(f"{dt.c_name} {name} = {expr};")
        self.scopes[-1][u] = name
        return name

    def expr(self, u):
        name = self.lookup(u)
        if name is not None:
            return name
        op = u.op
        if op is Ops.CONST:
            return c_literal(u.dtype, u.arg)
        if op is Ops.RANGE:
            return f"r{u.arg}"
        if op is Ops.SPECIAL:
            return u.arg[1]
        if op is Ops.DEFINE_ACC:
            return f"acc{u.arg}"
        if op is Ops.PARAM:
            return f"data{u.arg[1]}"
        if op is Ops.LOAD:
            idx_node = u.src[0]
            p = idx_node.src[0]
            idx = self.expr(idx_node.src[1])
            return self.new_var(u, f"data{p.arg[1]}[{idx}]")
        if op is Ops.WHERE:
            c, a, b = u.src
            return self.new_var(u, f"({self.expr(c)} ? {self.expr(a)} : {self.expr(b)})")
        if op is Ops.CAST:
            x = u.src[0]
            return self.new_var(u, f"({u.dtype.c_name})({self.expr(x)})")
        if op is Ops.RECIP:
            return self.new_var(u, f"(1.0f/{self.expr(u.src[0])})")
        if op is Ops.EXP2:
            return self.new_var(u, f"exp2f({self.expr(u.src[0])})")
        if op is Ops.LOG2:
            return self.new_var(u, f"log2f({self.expr(u.src[0])})")
        if op is Ops.SQRT:
            return self.new_var(u, f"sqrtf({self.expr(u.src[0])})")
        if op is Ops.MAX:
            a, b = u.src
            ea, eb = self.expr(a), self.expr(b)
            if u.dtype is dtypes.float32:
                return self.new_var(u, f"fmaxf({ea}, {eb})")
            return self.new_var(u, f"({ea} > {eb} ? {ea} : {eb})")
        if op is Ops.IDIV:
            a, b = u.src
            amin, _ = bounds(a)
            ea, eb = self.expr(a), self.expr(b)
            if amin >= 0:
                return self.new_var(u, f"({ea} / {eb})")
            return self.new_var(u, f"fdiv({ea}, {eb})")
        if op is Ops.MOD:
            a, b = u.src
            amin, _ = bounds(a)
            ea, eb = self.expr(a), self.expr(b)
            if amin >= 0:
                return self.new_var(u, f"({ea} % {eb})")
            return self.new_var(u, f"fmod_floor({ea}, {eb})")
        if op in (Ops.ADD, Ops.MUL, Ops.CMPLT, Ops.AND):
            a, b = u.src
            sym = {Ops.ADD: "+", Ops.MUL: "*", Ops.CMPLT: "<", Ops.AND: "&&"}[op]
            return self.new_var(u, f"({self.expr(a)} {sym} {self.expr(b)})")
        if op is Ops.STORE:
            idx_node, val = u.src
            p = idx_node.src[0]
            idx = self.expr(idx_node.src[1])
            self.emit(f"data{p.arg[1]}[{idx}] = {self.expr(val)};")
            return None
        raise NotImplementedError(f"expr for {op}")


def signature(name, ps, written):
    parts = []
    for p in ps:
        is_out = p.arg[1] in written
        const = "" if is_out else "const "
        parts.append(f"{const}{p.dtype.c_name}* restrict data{p.arg[1]}")
    return f"void {name}({', '.join(parts)})"


def render_sink(s, name="kernel"):
    ps = params_of(s)
    written = set()
    for st in s.src:
        p = st.src[0].src[0]
        written.add(p.arg[1])
    r = CRenderer()
    r.push()
    for st in s.src:
        r.expr(st)
    lines = [signature(name, ps, written) + " {"]
    lines.extend(r.lines)
    lines.append("}")
    return "\n".join(lines) + "\n"