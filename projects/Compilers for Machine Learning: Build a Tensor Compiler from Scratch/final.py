"""
Compilers for Machine Learning: Build a Tensor Compiler from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  UOp ──
import sys
import math
from enum import Enum, auto
import numpy as np

sys.setrecursionlimit(100000)
INF = float("inf")


class DType:
    def __init__(self, name, c_name, np_type):
        self.name, self.c_name, self.np = name, c_name, np_type
    def __repr__(self): return "dtypes." + self.name


class dtypes:
    bool = DType("bool", "int", np.int32)
    int32 = DType("int32", "int", np.int32)
    float32 = DType("float32", "float", np.float32)


class Ops(Enum):
    CONST = auto(); PARAM = auto(); BUFFER = auto()
    ADD = auto(); MUL = auto(); MAX = auto(); CMPLT = auto(); AND = auto(); IDIV = auto(); MOD = auto()
    RECIP = auto(); EXP2 = auto(); LOG2 = auto(); SQRT = auto(); CAST = auto(); WHERE = auto()
    RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto(); FLIP = auto(); PAD = auto(); SHRINK = auto(); REDUCE_AXIS = auto()
    RANGE = auto(); SPECIAL = auto(); INDEX = auto(); LOAD = auto(); STORE = auto(); DEFINE_ACC = auto(); SINK = auto()


BINARY = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPLT, Ops.AND, Ops.IDIV, Ops.MOD}
UNARY = {Ops.RECIP, Ops.EXP2, Ops.LOG2, Ops.SQRT, Ops.CAST}
ALU = BINARY | UNARY | {Ops.WHERE}
MOVEMENT = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.FLIP, Ops.PAD, Ops.SHRINK}


def _coerce(dtype, v):
    if dtype is dtypes.bool:
        return bool(v)
    if dtype is dtypes.int32:
        return int(v)
    if dtype is dtypes.float32:
        return float(v)
    return v


class UOp:
    __slots__ = ("op", "dtype", "src", "arg")
    _cache = {}

    def __new__(cls, op, dtype=None, src=(), arg=None):
        if not isinstance(op, Ops):
            op = Ops[op] if isinstance(op, str) else op
        src = tuple(src)
        key = (op, dtype, src, arg)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        node = super().__new__(cls)
        node.op = op
        node.dtype = dtype
        node.src = src
        node.arg = arg
        cls._cache[key] = node
        return node

    def __repr__(self):
        src_str = "(" + ", ".join(repr(s) for s in self.src) + ")"
        if self.arg is None:
            return f"UOp({self.op.name}, {self.dtype}, {src_str})"
        return f"UOp({self.op.name}, {self.dtype}, {src_str}, arg={self.arg})"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @staticmethod
    def const(dtype, v):
        return UOp(Ops.CONST, dtype, (), _coerce(dtype, v))

    @staticmethod
    def range(n, i):
        return UOp(Ops.RANGE, dtypes.int32, (UOp.const(dtypes.int32, n),), i)

    def alu(self, op, *src):
        if not src:
            src = (self,)
        new_src = tuple(
            UOp.const(self.dtype, s) if isinstance(s, (int, float, bool)) and not isinstance(s, UOp) else s
            for s in src
        )
        out_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.AND) else self.dtype
        return UOp(op, out_dtype, new_src)

    def __add__(self, other):
        return self.alu(Ops.ADD, self, other)

    def __radd__(self, other):
        return self.alu(Ops.ADD, other, self)

    def __mul__(self, other):
        return self.alu(Ops.MUL, self, other)

    def __rmul__(self, other):
        return self.alu(Ops.MUL, other, self)

    def __floordiv__(self, other):
        return self.alu(Ops.IDIV, self, other)

    def __mod__(self, other):
        return self.alu(Ops.MOD, self, other)

    def __lt__(self, other):
        return self.alu(Ops.CMPLT, self, other)

    def __and__(self, other):
        return self.alu(Ops.AND, self, other)

    def __neg__(self):
        if self.op is Ops.CONST:
            return UOp.const(self.dtype, -self.arg)
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def maximum(self, other):
        return self.alu(Ops.MAX, self, other)

    def recip(self):
        return self.alu(Ops.RECIP, self)

    def exp2(self):
        return self.alu(Ops.EXP2, self)

    def log2(self):
        return self.alu(Ops.LOG2, self)

    def sqrt(self):
        return self.alu(Ops.SQRT, self)

    def cast(self, dtype):
        return UOp(Ops.CAST, dtype, (self,))

    def where(self, a, b):
        if isinstance(a, UOp) and isinstance(b, UOp):
            wrap_dtype = a.dtype
        elif isinstance(a, UOp):
            wrap_dtype = a.dtype
        elif isinstance(b, UOp):
            wrap_dtype = b.dtype
        else:
            wrap_dtype = dtypes.float32
        a_w = UOp.const(wrap_dtype, a) if isinstance(a, (int, float, bool)) and not isinstance(a, UOp) else a
        b_w = UOp.const(wrap_dtype, b) if isinstance(b, (int, float, bool)) and not isinstance(b, UOp) else b
        return UOp(Ops.WHERE, a_w.dtype, (self, a_w, b_w))

    def toposort(self):
        order = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for s in reversed(node.src):
                if s not in visited:
                    stack.append((s, False))
        return order

# ── Step 002  bounds ──
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

# ── Step 003  UPat ──
_MATCH_ANY = object()


class UPat:
    def __init__(self, op=None, dtype=None, src=None, arg=_MATCH_ANY, name=None):
        self.op = op
        self.dtype = dtype
        self.src = src
        self.arg = arg
        self.name = name
        self._has_arg = arg is not _MATCH_ANY

    @staticmethod
    def var(name=None, dtype=None):
        return UPat(op=None, dtype=dtype, src=None, name=name)

    @staticmethod
    def cvar(name=None, dtype=None):
        return UPat(op=Ops.CONST, dtype=dtype, src=(), name=name)

    def match(self, u, store):
        if self.op is not None:
            if isinstance(self.op, (set, frozenset, list, tuple)):
                if u.op not in self.op:
                    return False
            elif u.op is not self.op:
                return False
        if self.dtype is not None and u.dtype is not self.dtype:
            return False
        if self._has_arg and u.arg != self.arg:
            return False
        if self.src is not None:
            if len(self.src) != len(u.src):
                return False
            for p, s in zip(self.src, u.src):
                if not p.match(s, store):
                    return False
        if self.name is not None:
            if self.name in store:
                if store[self.name] is not u:
                    return False
            else:
                store[self.name] = u
        return True


class PatternMatcher:
    def __init__(self, patterns):
        self.patterns = list(patterns)

    def __add__(self, other):
        return PatternMatcher(self.patterns + list(other.patterns))

    def rewrite(self, u):
        for pat, fn in self.patterns:
            store = {}
            if pat.match(u, store):
                result = fn(**store)
                if result is not None and result is not u:
                    return result
        return None

# ── Step 004  graph_rewrite ──
def graph_rewrite(root, pm):
    memo = {}

    def rw(u):
        if u in memo:
            return memo[u]

        new_src = tuple(rw(s) for s in u.src)
        if new_src != u.src:
            node = UOp(u.op, u.dtype, new_src, u.arg)
        else:
            node = u

        iters = 0
        while True:
            iters += 1
            if iters > 1000:
                break

            r = pm.rewrite(node)
            if r is None or r is node:
                break

            if r.src:
                r = rw(r)
            if r is node:
                break
            node = r
        
        memo[u] = node
        return node

    return rw(root)

# ── Step 005  exec_alu ──
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

# ── Step 006  fold_div ──
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

# ── Step 007  make_symbolic ──
def make_symbolic():
    rules = list(basic_rules().patterns)

    def fn_fold_div(u):
        x, c = u.src
        if c.op is Ops.CONST:
            return fold_div(x, c)
        return None
    rules.append((UPat(Ops.IDIV, dtype=dtypes.int32, src=(UPat.var(), UPat.cvar()), name="u"), fn_fold_div))

    def fn_fold_mod(u):
        x, c = u.src
        if c.op is Ops.CONST:
            return fold_mod(x, c)
        return None
    rules.append((UPat(Ops.MOD, dtype=dtypes.int32, src=(UPat.var(), UPat.cvar()), name="u"), fn_fold_mod))

    def fn_fold_cmplt(u):
        x, y = u.src
        return fold_cmplt(x, y)
    rules.append((UPat(Ops.CMPLT, dtype=dtypes.bool, src=(UPat(dtype=dtypes.int32), UPat.var()), name="u"), fn_fold_cmplt))

    def fn_fold_max(u):
        x, y = u.src
        return fold_max(x, y)
    rules.append((UPat(Ops.MAX, dtype=dtypes.int32, src=(UPat(dtype=dtypes.int32), UPat.var()), name="u"), fn_fold_max))

    return PatternMatcher(rules)


symbolic = make_symbolic()


def simplify(u):
    return graph_rewrite(u, symbolic)

# ── Step 008  render_sink ──
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

# ── Step 009  eval_sink ──
import numpy as np

def eval_sink(s, bufs):
    memo = {}

    def ev(u):
        if u in memo:
            return memo[u]
        op = u.op
        if op is Ops.CONST:
            val = u.arg
        elif op is Ops.LOAD:
            idx_node = u.src[0]
            p = idx_node.src[0]
            idx = ev(idx_node.src[1])
            val = bufs[p.arg[0]].reshape(-1)[idx].item()
        elif op in ALU:
            vals = [ev(x) for x in u.src]
            val = exec_alu(op, u.dtype, vals).arg
        elif op is Ops.CAST:
            val = ev(u.src[0])
        else:
            raise NotImplementedError(f"eval for {op}")
        memo[u] = val
        return val

    for st in s.src:
        idx_node, val_node = st.src
        p = idx_node.src[0]
        idx = ev(idx_node.src[1])
        val = ev(val_node)
        bufs[p.arg[0]].reshape(-1)[idx] = val

    return bufs

# ── Step 010  compile_c ──
import os
import shutil
import tempfile
import hashlib
import ctypes


_compiler = None
_libs = {}


def compiler():
    global _compiler
    if _compiler is None:
        env = os.environ.get("CC")
        if env:
            _compiler = env
        else:
            for name in ("clang", "gcc", "cc"):
                path = shutil.which(name)
                if path:
                    _compiler = path
                    break
            if _compiler is None:
                raise RuntimeError("no C compiler found")
    return _compiler


def compile_c(src):
    full_src = C_HEADER + "\n" + src
    key = hashlib.sha1(full_src.encode()).hexdigest()
    if key in _libs:
        return _libs[key]
    tmpdir = tempfile.mkdtemp()
    c_path = os.path.join(tmpdir, "kernel.c")
    so_path = os.path.join(tmpdir, "kernel.so")
    with open(c_path, "w") as f:
        f.write(full_src)
    import subprocess
    cmd = [compiler(), "-O3", "-march=native", "-shared", "-fPIC", "-w",
           c_path, "-o", so_path, "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    lib = ctypes.CDLL(so_path)
    _libs[key] = lib
    return lib


def call_kernel(lib, name, bufs):
    fn = getattr(lib, name)
    args = [ctypes.c_void_p(buf.ctypes.data) for buf in bufs]
    fn(*args)


def run_sink(s, bufs, name="kernel"):
    src = render_sink(s, name)
    lib = compile_c(src)
    ps = params_of(s)
    ordered = [bufs[p.arg[0]] for p in ps]
    call_kernel(lib, name, ordered)
    return bufs

# ── Step 011  shape_of ──
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
        if not u.src:
            return ()
        if op is Ops.WHERE:
            return shape_of(u.src[1])
        return shape_of(u.src[0])
    raise NotImplementedError(f"shape_of for {op}")

# ── Step 012  Tensor ──
import math


class Tensor:
    def __init__(self, uop):
        self.uop = uop

    @staticmethod
    def input(name, shape, dtype=dtypes.float32):
        return Tensor(buffer(name, shape, dtype))

    @staticmethod
    def const(v, dtype=dtypes.float32):
        return Tensor(UOp.const(dtype, v))

    @property
    def shape(self):
        return shape_of(self.uop)

    @property
    def dtype(self):
        return self.uop.dtype

    def __repr__(self):
        return f"Tensor(shape={self.shape}, {self.dtype})"

    # ---- movement ops ----
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        # resolve -1
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            total = 1
            for s in self.shape:
                total *= s
            resolved = total // known
            shape = tuple(resolved if s == -1 else s for s in shape)
        total = 1
        for s in self.shape:
            total *= s
        new_total = 1
        for s in shape:
            new_total *= s
        assert total == new_total, f"cannot reshape {self.shape} to {shape}"
        if shape == self.shape:
            return self
        # collapse consecutive reshapes
        src = self.uop.src[0] if self.uop.op is Ops.RESHAPE else self.uop
        return Tensor(UOp(Ops.RESHAPE, self.dtype, (src,), shape))

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        assert len(shape) == len(self.shape), "expand requires same rank"
        for old, new in zip(self.shape, shape):
            assert old == new or old == 1, f"cannot expand {self.shape} to {shape}"
        if shape == self.shape:
            return self
        return Tensor(UOp(Ops.EXPAND, self.dtype, (self.uop,), shape))

    def permute(self, *perm):
        if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
            perm = tuple(perm[0])
        else:
            perm = tuple(perm)
        if perm == tuple(range(len(self.shape))):
            return self
        return Tensor(UOp(Ops.PERMUTE, self.dtype, (self.uop,), perm))

    def flip(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        ndim = len(self.shape)
        norm = sorted(set(a % ndim for a in axes))
        if not norm:
            return self
        return Tensor(UOp(Ops.FLIP, self.dtype, (self.uop,), tuple(norm)))

    def pad(self, pads):
        pads = tuple((int(lo), int(hi)) for lo, hi in pads)
        if all(lo == 0 and hi == 0 for lo, hi in pads):
            return self
        return Tensor(UOp(Ops.PAD, self.dtype, (self.uop,), pads))

    def shrink(self, ranges):
        ranges = tuple((int(b), int(e)) for b, e in ranges)
        for (b, e), s in zip(ranges, self.shape):
            assert 0 <= b < e <= s, f"bad shrink {ranges} for shape {self.shape}"
        if all(b == 0 and e == s for (b, e), s in zip(ranges, self.shape)):
            return self
        return Tensor(UOp(Ops.SHRINK, self.dtype, (self.uop,), ranges))

    def transpose(self, a=-2, b=-1):
        ndim = len(self.shape)
        a %= ndim
        b %= ndim
        perm = list(range(ndim))
        perm[a], perm[b] = perm[b], perm[a]
        return self.permute(*perm)

    # ---- broadcasting ----
    def _bcast(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        a, b = self, other
        while len(a.shape) < len(b.shape):
            a = a.reshape((1,) + a.shape)
        while len(b.shape) < len(a.shape):
            b = b.reshape((1,) + b.shape)
        target = tuple(max(x, y) for x, y in zip(a.shape, b.shape))
        a = a.expand(*target)
        b = b.expand(*target)
        return a, b

    def _alu(self, op, *others):
        ts = [self]
        for o in others:
            if not isinstance(o, Tensor):
                o = Tensor.const(o, self.dtype)
            ts.append(o)
        # broadcast all pairwise against the running shape
        while len(ts) > 1:
            a = ts.pop(0)
            b = ts.pop(0)
            a, b = a._bcast(b)
            ts.insert(0, a)
            ts.insert(1, b)
            # now a and b share shape; combine into one node
            if op is Ops.CMPLT or op is Ops.AND:
                dt = dtypes.bool
            elif op is Ops.WHERE:
                dt = ts[1].dtype
            else:
                dt = a.dtype
            # for WHERE, keep 3 sources
            if op is Ops.WHERE:
                return None  # handled separately
            node = UOp(op, dt, (a.uop, b.uop))
            ts = [Tensor(node)] + ts[2:]
            break
        return ts[0]

    # ---- elementwise ----
    def _binary(self, op, other, dtype=None):
        a = self
        b = other if isinstance(other, Tensor) else Tensor.const(other, self.dtype)
        a, b = a._bcast(b)
        if op is Ops.CMPLT or op is Ops.AND:
            dt = dtypes.bool
        elif op is Ops.WHERE:
            dt = a.dtype
        else:
            dt = a.dtype if dtype is None else dtype
        return Tensor(UOp(op, dt, (a.uop, b.uop)))

    def __add__(self, other):
        return self._binary(Ops.ADD, other)

    def __radd__(self, other):
        return self._binary(Ops.ADD, other)

    def __mul__(self, other):
        return self._binary(Ops.MUL, other)

    def __rmul__(self, other):
        return self._binary(Ops.MUL, other)

    def __neg__(self):
        return self._binary(Ops.MUL, -1.0)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        return other + (-self)

    def recip(self):
        return Tensor(UOp(Ops.RECIP, self.dtype, (self.uop,)))

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        return self * other.recip()

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        return other * self.recip()

    def __floordiv__(self, other):
        return self._binary(Ops.IDIV, other)

    def __rfloordiv__(self, other):
        return self._binary(Ops.IDIV, other)

    def __mod__(self, other):
        return self._binary(Ops.MOD, other)

    def __rmod__(self, other):
        return self._binary(Ops.MOD, other)

    def __lt__(self, other):
        return self._binary(Ops.CMPLT, other)

    def __gt__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)
        return other < self

    def __and__(self, other):
        return self._binary(Ops.AND, other)

    def maximum(self, other):
        return self._binary(Ops.MAX, other)

    def where(self, a, b):
        cond = self
        if not isinstance(a, Tensor):
            a = Tensor.const(a, self.dtype if not isinstance(b, Tensor) else b.dtype)
        if not isinstance(b, Tensor):
            b = Tensor.const(b, a.dtype)
        cond, a = cond._bcast(a)
        cond, b = cond._bcast(b)
        a, b = a._bcast(b)
        return Tensor(UOp(Ops.WHERE, a.dtype, (cond.uop, a.uop, b.uop)))

    def exp2(self):
        return Tensor(UOp(Ops.EXP2, self.dtype, (self.uop,)))

    def log2(self):
        return Tensor(UOp(Ops.LOG2, self.dtype, (self.uop,)))

    def sqrt(self):
        return Tensor(UOp(Ops.SQRT, self.dtype, (self.uop,)))

    def exp(self):
        return (self * (1.0 / math.log(2))).exp2()

    def log(self):
        return self.log2() * math.log(2)

    def relu(self):
        return self.maximum(0.0)

    def square(self):
        return self * self

    def cast(self, dtype):
        return Tensor(UOp(Ops.CAST, dtype, (self.uop,)))

    # ---- reductions ----
    def _reduce(self, op, axis, keepdim):
        ndim = len(self.shape)
        if axis is None:
            axes = tuple(range(ndim))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            axes = tuple(sorted(set(a % ndim for a in axis)))
        node = UOp(Ops.REDUCE_AXIS, self.dtype, (self.uop,), (op, axes))
        if not keepdim:
            new_shape = tuple(1 if i in axes else s for i, s in enumerate(self.shape))
            new_shape = tuple(s for i, s in enumerate(new_shape) if i not in axes)
            node = UOp(Ops.RESHAPE, self.dtype, (node,), new_shape)
        return Tensor(node)

    def sum(self, axis=None, keepdim=False):
        return self._reduce(Ops.ADD, axis, keepdim)

    def max(self, axis=None, keepdim=False):
        return self._reduce(Ops.MAX, axis, keepdim)

    def mean(self, axis=None, keepdim=False):
        s = self.sum(axis, keepdim)
        ndim = len(self.shape)
        if axis is None:
            count = 1
            for d in self.shape:
                count *= d
        else:
            if isinstance(axis, int):
                axis = (axis,)
            count = 1
            for a in axis:
                count *= self.shape[a % ndim]
        return s * (1.0 / count)

    # ---- composites ----
    def matmul(self, w):
        m, k = self.shape[-2], self.shape[-1]
        n = w.shape[-1]
        a = self.reshape(*self.shape[:-2], m, 1, k)
        wt = w.transpose(-1, -2).reshape(*w.shape[:-2], 1, n, k)
        return (a * wt).sum(-1)

    def __matmul__(self, w):
        return self.matmul(w)

    def softmax(self, axis=-1):
        m = self.max(axis, keepdim=True)
        e = (self - m).exp()
        return e / e.sum(axis, keepdim=True)

    def logsoftmax(self, axis=-1):
        m = self.max(axis, keepdim=True)
        z = self - m
        return z - z.exp().sum(axis, keepdim=True).log()

    def layernorm(self, eps=1e-5):
        mean = self.mean(-1, keepdim=True)
        centered = self - mean
        var = centered.square().mean(-1, keepdim=True)
        return centered / (var + eps).sqrt()

# ── Step 013  eval_tensor ──
import numpy as np

def eval_tensor(u, bufs, cache=None):
    if cache is None:
        cache = {}
    if u in cache:
        return cache[u]

    op = u.op

    if op is Ops.BUFFER:
        name, shape = u.arg
        arr = np.asarray(bufs[name]).astype(u.dtype.np).reshape(shape)
        cache[u] = arr
        return arr
    if op is Ops.CONST:
        arr = np.asarray(u.arg).astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op is Ops.RESHAPE:
        src = eval_tensor(u.src[0], bufs, cache)
        arr = src.reshape(u.arg).astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op is Ops.EXPAND:
        src = eval_tensor(u.src[0], bufs, cache)
        arr = np.broadcast_to(src, u.arg).astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op is Ops.PERMUTE:
        src = eval_tensor(u.src[0], bufs, cache)
        arr = np.transpose(src, u.arg).astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op is Ops.FLIP:
        src = eval_tensor(u.src[0], bufs, cache)
        arr = np.flip(src, axis=u.arg).astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op is Ops.PAD:
        src = eval_tensor(u.src[0], bufs, cache)
        arr = np.pad(src, u.arg).astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op is Ops.SHRINK:
        src = eval_tensor(u.src[0], bufs, cache)
        slices = tuple(slice(b, e) for b, e in u.arg)
        arr = src[slices].astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op is Ops.REDUCE_AXIS:
        src = eval_tensor(u.src[0], bufs, cache)
        rop, axes = u.arg
        with np.errstate(all="ignore"):
            if rop is Ops.ADD:
                arr = np.sum(src, axis=axes, keepdims=True)
            elif rop is Ops.MAX:
                arr = np.max(src, axis=axes, keepdims=True)
            else:
                raise NotImplementedError(f"reduce {rop}")
        arr = arr.astype(u.dtype.np)
        cache[u] = arr
        return arr
    if op in ALU:
        vals = [eval_tensor(s, bufs, cache) for s in u.src]
        with np.errstate(all="ignore"):
            if op is Ops.ADD:
                arr = vals[0] + vals[1]
            elif op is Ops.MUL:
                arr = vals[0] * vals[1]
            elif op is Ops.MAX:
                arr = np.maximum(vals[0], vals[1])
            elif op is Ops.CMPLT:
                arr = vals[0] < vals[1]
            elif op is Ops.AND:
                arr = vals[0] & vals[1]
            elif op is Ops.IDIV:
                arr = np.floor_divide(vals[0], vals[1])
            elif op is Ops.MOD:
                arr = np.mod(vals[0], vals[1])
            elif op is Ops.RECIP:
                arr = np.reciprocal(vals[0])
            elif op is Ops.EXP2:
                arr = np.exp2(vals[0])
            elif op is Ops.LOG2:
                arr = np.log2(vals[0])
            elif op is Ops.SQRT:
                arr = np.sqrt(vals[0])
            elif op is Ops.CAST:
                arr = vals[0]
            elif op is Ops.WHERE:
                arr = np.where(vals[0], vals[1], vals[2])
            else:
                raise NotImplementedError(f"alu {op}")
        arr = np.asarray(arr).astype(u.dtype.np)
        cache[u] = arr
        return arr

    raise NotImplementedError(f"eval_tensor for {op}")

# ── Step 014  conv2d ──
def conv2d(x, w, pad=0):
    N, Cin, H, W = x.shape
    Cout, _, kh, kw = w.shape
    Ho = H + 2 * pad - kh + 1
    Wo = W + 2 * pad - kw + 1
    
    if pad > 0:
        xp = x.pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))
    else:
        xp = x
    
    total = None
    for dh in range(kh):
        for dw in range(kw):
            win = xp.shrink(((0, N), (0, Cin), (dh, dh + Ho), (dw, dw + Wo)))
            win = win.reshape(N, 1, Cin, Ho, Wo).expand(N, Cout, Cin, Ho, Wo)
            wt = w.shrink(((0, Cout), (0, Cin), (dh, dh + 1), (dw, dw + 1)))
            wt = wt.reshape(1, Cout, Cin, 1, 1).expand(N, Cout, Cin, Ho, Wo)
            prod = win * wt
            total = prod if total is None else total + prod
    
    return total.sum(2)

Tensor.conv2d = conv2d

# ── Step 015  Kernel ──
class Reduce:
    def __init__(self, ranges, accs, body):
        self.ranges = tuple(ranges)
        self.accs = [list(a) for a in accs]
        self.body = list(body)

    def __repr__(self):
        return f"Reduce(ranges={[r.arg for r in self.ranges]}, accs={len(self.accs)}, body={self.body!r})"

    def map_exprs(self, f):
        new_accs = [[a[0], f(a[1]), f(a[2])] for a in self.accs]
        new_body = [b.map_exprs(f) if isinstance(b, Reduce) else b for b in self.body]
        return Reduce(self.ranges, new_accs, new_body)

    def all_exprs(self):
        out = []
        for a in self.accs:
            out.append(a[1])
            out.append(a[2])
        for b in self.body:
            if isinstance(b, Reduce):
                out.extend(b.all_exprs())
        return out


class Kernel:
    def __init__(self, name, params, out_ranges, body, stores):
        self.name = name
        self.params = list(params)
        self.out_ranges = list(out_ranges)
        self.body = list(body)
        self.stores = list(stores)

    def map_exprs(self, f):
        new_body = [b.map_exprs(f) if isinstance(b, Reduce) else b for b in self.body]
        new_stores = [(f(i), f(v)) for i, v in self.stores]
        return Kernel(self.name, self.params, self.out_ranges, new_body, new_stores)

    def all_exprs(self):
        out = []
        for b in self.body:
            if isinstance(b, Reduce):
                out.extend(b.all_exprs())
        for i, v in self.stores:
            out.append(i)
            out.append(v)
        return out


def ranges_in(u):
    result = set()
    visited = set()
    stack = [u]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op is Ops.RANGE:
            result.add(node)
        for s in node.src:
            stack.append(s)
    return result


def flat_index(idxs, shape):
    strides = strides_of(shape)
    total = UOp.const(dtypes.int32, 0)
    for idx, stride in zip(idxs, strides):
        if stride == 1:
            term = idx
        else:
            term = idx * stride
        total = total + term
    return total


def load(p, idx):
    if isinstance(idx, UOp):
        idx_node = idx
    else:
        idx_node = UOp.const(dtypes.int32, idx)
    return UOp(Ops.LOAD, p.dtype, (UOp(Ops.INDEX, dtypes.int32, (p, idx_node)),))

# ── Step 016  Lowerer ──
class Lowerer:
    def __init__(self, realized):
        self.realized = realized
        self.n = 0
        self.memo = {}
        self.frames = []

    def new_range(self, n):
        r = UOp.range(n, self.n)
        self.n += 1
        return r

    def new_acc(self, dtype):
        a = UOp(Ops.DEFINE_ACC, dtype, (), self.n)
        self.n += 1
        return a

    def index(self, u, idxs, root):
        key = (u, idxs)
        if key in self.memo:
            return self.memo[key]
        op = u.op

        if u in self.realized and u is not root:
            val = load(self.realized[u], flat_index(idxs, shape_of(u)))
        elif op is Ops.BUFFER:
            val = load(self.realized[u], flat_index(idxs, shape_of(u)))
        elif op is Ops.CONST:
            val = u
        elif op in ALU:
            vals = [self.index(s, idxs, root) for s in u.src]
            dt = dtypes.bool if op in (Ops.CMPLT, Ops.AND) else u.dtype
            val = UOp(op, dt, tuple(vals))
        elif op is Ops.RESHAPE:
            f = flat_index(idxs, shape_of(u))
            src = u.src[0]
            if src in self.realized or src.op is Ops.BUFFER:
                val = load(self.realized[src], f)
            else:
                src_shape = shape_of(src)
                strides = strides_of(src_shape)
                src_idx = []
                for j, (s, stride) in enumerate(zip(src_shape, strides)):
                    if s == 1:
                        src_idx.append(UOp.const(dtypes.int32, 0))
                    else:
                        src_idx.append((f // stride) % s)
                val = self.index(src, tuple(src_idx), root)
        elif op is Ops.EXPAND:
            src = u.src[0]
            src_shape = shape_of(src)
            src_idx = tuple(
                UOp.const(dtypes.int32, 0) if s == 1 else idxs[k]
                for k, s in enumerate(src_shape)
            )
            val = self.index(src, src_idx, root)
        elif op is Ops.PERMUTE:
            perm = u.arg
            src_idx = [None] * len(perm)
            for k, p in enumerate(perm):
                src_idx[p] = idxs[k]
            val = self.index(u.src[0], tuple(src_idx), root)
        elif op is Ops.FLIP:
            src_shape = shape_of(u.src[0])
            axes = set(u.arg)
            src_idx = tuple(
                (UOp.const(dtypes.int32, src_shape[k] - 1) - idxs[k]) if k in axes else idxs[k]
                for k in range(len(src_shape))
            )
            val = self.index(u.src[0], src_idx, root)
        elif op is Ops.SHRINK:
            ranges = u.arg
            src_idx = tuple(idxs[k] + b for k, (b, e) in enumerate(ranges))
            val = self.index(u.src[0], src_idx, root)
        elif op is Ops.PAD:
            pads = u.arg
            src_shape = shape_of(u.src[0])
            src_idx = []
            valid = UOp.const(dtypes.bool, True)
            for k, (lo, hi) in enumerate(pads):
                src_idx.append(idxs[k] - lo)
                if lo > 0:
                    valid = valid & (UOp.const(dtypes.int32, lo - 1) < idxs[k])
                if hi > 0:
                    valid = valid & (idxs[k] < UOp.const(dtypes.int32, lo + src_shape[k]))
            inner = self.index(u.src[0], tuple(src_idx), root)
            val = valid.where(inner, UOp.const(u.dtype, 0))
        elif op is Ops.REDUCE_AXIS:
            raise NotImplementedError("REDUCE_AXIS lowering is the next step")
        else:
            raise NotImplementedError(f"index for {op}")

        self.memo[key] = val
        return val

    def lower(self, root):
        shape = shape_of(root)
        out_ranges = []
        idxs = []
        for d in shape:
            if d == 1:
                idxs.append(UOp.const(dtypes.int32, 0))
            else:
                r = self.new_range(d)
                out_ranges.append(r)
                idxs.append(r)
        idxs = tuple(idxs)
        self.frames.append((set(out_ranges), []))
        val = self.index(root, idxs, root)
        frame = self.frames.pop()
        body = frame[1]
        store_idx = flat_index(idxs, shape)
        return Kernel("k", [], out_ranges, body, [(store_idx, val)])


def lower_kernel(root, inputs, name="k"):
    realized = {}
    params = []
    out_param = param("out", root.dtype, 0)
    for i, (node, bufname) in enumerate(inputs.items()):
        realized[node] = param(bufname, node.dtype, i + 1)
    lowerer = Lowerer(realized)
    k = lowerer.lower(root)
    k.name = name
    k.params = [out_param] + [realized[n] for n in inputs]
    return k.map_exprs(simplify)

# ── Step 017  ReduceLowerer ──
def reduce_exprs(r):
    out = []
    for a in r.accs:
        out.append(a[1])
        out.append(a[2])
    for b in r.body:
        out.extend(reduce_exprs(b))
    return out


class ReduceLowerer(Lowerer):
    def index(self, u, idxs, root):
        if u.op is not Ops.REDUCE_AXIS or (u in self.realized and u is not root):
            return super().index(u, idxs, root)

        key = (u, idxs)
        if key in self.memo:
            return self.memo[key]

        op, axes = u.arg
        src = u.src[0]
        src_shape = shape_of(src)

        new_ranges = []
        src_idx = list(idxs)
        for ax in axes:
            r = self.new_range(src_shape[ax])
            new_ranges.append(r)
            src_idx[ax] = r
        src_idx = tuple(src_idx)

        enclosing = set()
        for frame in self.frames:
            enclosing |= frame[0]
        self.frames.append((enclosing | set(new_ranges), []))
        val = self.index(src, src_idx, root)
        frame = self.frames.pop()
        body = frame[1]

        acc = self.new_acc(u.dtype)
        if op is Ops.ADD:
            init = UOp.const(u.dtype, 0.0)
            update = acc + val
        elif op is Ops.MAX:
            init = UOp.const(u.dtype, -INF)
            update = acc.maximum(val)
        else:
            raise NotImplementedError(f"reduce op {op}")

        stmt = Reduce(new_ranges, [[acc, init, update]], body)

        deps = set()
        for e in [init, update] + reduce_exprs(stmt):
            deps |= ranges_in(e)
        deps -= set(new_ranges)

        target = None
        for frame in self.frames:
            if deps <= frame[0]:
                target = frame
        if target is None:
            target = self.frames[-1]
        target[1].append(stmt)

        self.memo[key] = acc
        return acc


def lower_kernel(root, inputs, name="k"):
    realized = {}
    out_param = param("out", root.dtype, 0)
    for i, (node, bufname) in enumerate(inputs.items()):
        realized[node] = param(bufname, node.dtype, i + 1)
    lowerer = ReduceLowerer(realized)
    k = lowerer.lower(root)
    k.name = name
    k.params = [out_param] + [realized[n] for n in inputs]
    return k.map_exprs(simplify)

# ── Step 018  run_kernel_np ──
def run_kernel_np(k, bufs):
    def rsize(r):
        if r.op in (Ops.RANGE, Ops.SPECIAL):
            return int(r.src[0].arg)
        return 1

    shape = tuple(rsize(r) for r in k.out_ranges)

    if shape:
        grid = np.indices(shape)
    else:
        grid = np.empty((0,), dtype=np.int64)

    env = {}
    for i, r in enumerate(k.out_ranges):
        if r.op in (Ops.RANGE, Ops.SPECIAL):
            env[r] = grid[i]
        # CONST output "ranges" evaluate to their constant via ev; no binding needed.

    out = bufs[0]
    out_flat = out.reshape(-1)

    def ev(u, cache):
        if u in cache:
            return cache[u]
        op = u.op
        if op is Ops.CONST:
            val = np.float32(u.arg) if u.dtype is dtypes.float32 else np.int64(u.arg)
        elif op in (Ops.RANGE, Ops.DEFINE_ACC, Ops.SPECIAL):
            val = env[u]
        elif op is Ops.LOAD:
            idx_node = u.src[0]
            p = idx_node.src[0]
            idx = ev(idx_node.src[1], cache)
            flat = bufs[int(p.arg[1])].reshape(-1)
            clipped = np.clip(idx, 0, flat.shape[0] - 1)
            val = flat[clipped]
        elif op is Ops.WHERE:
            c, a, b = u.src
            val = np.where(ev(c, cache), ev(a, cache), ev(b, cache))
        elif op is Ops.CAST:
            x = ev(u.src[0], cache)
            val = x.astype(np.float32) if u.dtype is dtypes.float32 else x.astype(np.int64)
        elif op in ALU:
            vals = [ev(s, cache) for s in u.src]
            with np.errstate(all="ignore"):
                if op is Ops.ADD: val = vals[0] + vals[1]
                elif op is Ops.MUL: val = vals[0] * vals[1]
                elif op is Ops.MAX: val = np.maximum(vals[0], vals[1])
                elif op is Ops.CMPLT: val = vals[0] < vals[1]
                elif op is Ops.AND: val = vals[0] & vals[1]
                elif op is Ops.IDIV: val = np.floor_divide(vals[0], vals[1])
                elif op is Ops.MOD: val = np.mod(vals[0], vals[1])
                elif op is Ops.RECIP: val = np.reciprocal(vals[0].astype(np.float32))
                elif op is Ops.EXP2: val = np.exp2(vals[0])
                elif op is Ops.LOG2: val = np.log2(vals[0])
                elif op is Ops.SQRT: val = np.sqrt(vals[0])
                else: raise NotImplementedError(op)
            val = np.asarray(val, dtype=np.float32) if u.dtype is dtypes.float32 else np.asarray(val).astype(np.int64)
        else:
            raise NotImplementedError(op)
        cache[u] = val
        return val

    def run_reduce(r):
        c0 = {}
        for acc, init, update in r.accs:
            v = ev(init, c0)
            env[acc] = np.broadcast_to(v, shape).copy() if shape else np.array(v)
        sizes = tuple(int(q.src[0].arg) for q in r.ranges)
        indices = list(np.ndindex(*sizes)) if sizes else [()]
        for idx in indices:
            for q, i in zip(r.ranges, idx):
                env[q] = np.int64(i)
            c = {}
            for nested in r.body:
                run_reduce(nested)
            updates = [(acc, ev(update, c)) for acc, init, update in r.accs]
            for acc, v in updates:
                env[acc] = v
            c.clear()

    for stmt in k.body:
        run_reduce(stmt)

    c = {}
    for idx_expr, val_expr in k.stores:
        idx = ev(idx_expr, c)
        val = ev(val_expr, c)
        flat_idx = np.asarray(idx).reshape(-1).astype(np.int64)
        flat_val = np.asarray(val, dtype=out.dtype).reshape(-1)
        out_flat[flat_idx] = flat_val

    return out

# ── Step 019  render_kernel ──
def render_kernel(k):
    r = CRenderer()

    def emit(line):
        level = len(r.scopes) - 1
        indent = "" if level <= 0 else "    " + "  " * (level - 1)
        r.lines.append(indent + line)

    r.emit = emit  # route new_var / expr-generated lines through the same indentation

    for p in k.params:
        r.scopes[0][p] = f"data{p.arg[1]}"

    r.push()

    out_ranges = []
    for rng in k.out_ranges:
        if rng.op is Ops.RANGE:
            i = rng.arg
            emit(f"for (int r{i} = 0; r{i} < {rng.src[0].arg}; r{i}++) {{")
            r.push()
            out_ranges.append(rng)

    def render_reduce(red):
        accs = []
        for acc, init, update in red.accs:
            i = acc.arg
            emit(f"{acc.dtype.c_name} acc{i} = {r.expr(init)};")
            r.scopes[-1][acc] = f"acc{i}"
            accs.append((acc, update, i))
        loop_ranges = []
        for rng in red.ranges:
            i = rng.arg
            emit(f"for (int r{i} = 0; r{i} < {rng.src[0].arg}; r{i}++) {{")
            r.push()
            r.scopes[-1][rng] = f"r{i}"
            loop_ranges.append(rng)
        for nested in red.body:
            render_reduce(nested)
        exprs = [r.expr(update) for acc, update, i in accs]
        for (acc, update, i), e in zip(accs, exprs):
            emit(f"acc{i} = {e};")
        for _ in loop_ranges:
            r.pop()
            emit("}")

    for stmt in k.body:
        render_reduce(stmt)

    for idx_expr, val_expr in k.stores:
        emit(f"data0[{r.expr(idx_expr)}] = {r.expr(val_expr)};")

    for _ in out_ranges:
        r.pop()
        emit("}")

    r.pop()

    sig = signature(k.name, k.params, {k.params[0].arg[1]})
    return "\n".join([sig + " {"] + r.lines + ["}"]) + "\n"


def run_kernel_c(k, bufs):
    lib = compile_c(render_kernel(k))
    call_kernel(lib, k.name, bufs)
    return bufs[0]

# ── Step 020  schedule ──
class Program:
    def __init__(self):
        self.buffers = {}
        self.kernels = []


def schedule(outputs):
    prog = Program()
    seen = set()
    order = []
    for name, t in outputs.items():
        for u in t.uop.toposort():
            if u not in seen:
                seen.add(u)
                order.append(u)

    realized_name = {}

    # 1) inputs
    for u in order:
        if u.op is Ops.BUFFER:
            bufname = u.arg[0]
            realized_name[u] = bufname
            prog.buffers[bufname] = (shape_of(u), u.dtype, "input")

    # 2) stages: every REDUCE_AXIS not already realized, numbered in toposort order
    stage_counter = 0
    for u in order:
        if u.op is Ops.REDUCE_AXIS and u not in realized_name:
            bufname = f"stage{stage_counter}"
            stage_counter += 1
            realized_name[u] = bufname
            prog.buffers[bufname] = (shape_of(u), u.dtype, "stage")

    # 3) outputs
    for name, t in outputs.items():
        realized_name[t.uop] = name
        prog.buffers[name] = (shape_of(t.uop), t.uop.dtype, "output")

    # kernels
    kernel_idx = 0
    for u in order:
        if u not in realized_name or u.op is Ops.BUFFER:
            continue
        out_name = realized_name[u]
        sub_seen = set()
        sub_order = []
        for node in u.toposort():
            if node is u:
                continue
            if node in realized_name and node not in sub_seen:
                sub_seen.add(node)
                sub_order.append(node)
        ins = {node: realized_name[node] for node in sub_order}
        k = lower_kernel(u, ins, name=f"k{kernel_idx}")
        kernel_idx += 1
        prog.kernels.append((k, [out_name] + [realized_name[n] for n in sub_order]))

    return prog

# ── Step 021  Compiled ──
import numpy as np


class Compiled:
    def __init__(self, prog):
        self.prog = prog

        # Render every kernel with placeholder name K, deduplicate sources.
        sources = []
        kernel_to_unique = []
        unique_of = {}
        for k, names in prog.kernels:
            src = render_kernel(k)
            placeholder = src.replace(f"void {k.name}(", "void K(")
            if placeholder not in unique_of:
                unique_of[placeholder] = len(sources)
                sources.append(placeholder)
            kernel_to_unique.append(unique_of[placeholder])

        self.names = [f"k{i}" for i in range(len(sources))]
        self.unique = len(sources)
        self.kernel_to_unique = kernel_to_unique

        # Replace the placeholder with the assigned names and compile all at once.
        parts = []
        for i, src in enumerate(sources):
            parts.append(src.replace("void K(", f"void {self.names[i]}(", 1))
        combined = "\n".join(parts)
        self.lib = compile_c(combined)

    def run(self, inputs):
        bufs = {}
        for name, (shape, dtype, kind) in self.prog.buffers.items():
            if kind == "input":
                arr = np.ascontiguousarray(inputs[name], dtype=dtype.np).reshape(shape)
                bufs[name] = arr
            else:
                bufs[name] = np.zeros(shape, dtype=dtype.np)

        for (k, names), uniq in zip(self.prog.kernels, self.kernel_to_unique):
            ordered = [bufs[n] for n in names]
            call_kernel(self.lib, self.names[uniq], ordered)

        return {name: bufs[name] for name, (_, _, kind) in self.prog.buffers.items()
                if kind == "output"}


def run_program_np(prog, inputs):
    bufs = {}
    for name, (shape, dtype, kind) in prog.buffers.items():
        if kind == "input":
            arr = np.ascontiguousarray(inputs[name], dtype=dtype.np).reshape(shape)
            bufs[name] = arr
        else:
            bufs[name] = np.zeros(shape, dtype=dtype.np)

    for k, names in prog.kernels:
        ordered = [bufs[n] for n in names]
        run_kernel_np(k, ordered)

    return {name: bufs[name] for name, (_, _, kind) in prog.buffers.items()
            if kind == "output"}

# ── Step 022  split_range ──
from functools import lru_cache


def substitute(u, mapping):
    memo = {}

    def rebuild(node):
        if node in memo:
            return memo[node]
        if node in mapping:
            result = mapping[node]
            memo[node] = result
            return result
        new_src = tuple(rebuild(s) for s in node.src)
        if new_src == node.src:
            result = node
        else:
            result = UOp(node.op, node.dtype, new_src, node.arg)
        memo[node] = result
        return result

    return rebuild(u)


def next_id(k):
    best = -1

    def visit_expr(e):
        nonlocal best
        for node in e.toposort():
            if node.op in (Ops.RANGE, Ops.DEFINE_ACC):
                best = max(best, node.arg)

    for r in k.out_ranges:
        if r.op is Ops.RANGE:
            best = max(best, r.arg)

    for e in k.all_exprs():
        visit_expr(e)

    return best + 1


def split_range(k, rng, size):
    n = int(rng.src[0].arg)
    assert n % size == 0, f"range size {n} not divisible by {size}"
    nid = next_id(k)
    outer = UOp.range(n // size, nid)
    inner = UOp.range(size, nid + 1)
    repl = outer * size + inner

    # substitute in every expression of the kernel (recursively through reduces)
    k2 = k.map_exprs(lambda e: simplify(substitute(e, {rng: repl})))

    # replace rng by outer, inner in out_ranges and in every reduce's ranges
    def fix_ranges(ranges):
        out = []
        for r in ranges:
            if r is rng:
                out.append(outer)
                out.append(inner)
            else:
                out.append(r)
        return tuple(out)

    k2.out_ranges = list(fix_ranges(k2.out_ranges))

    def fix_body(body):
        for red in body:
            red.ranges = fix_ranges(red.ranges)
            fix_body(red.body)

    fix_body(k2.body)

    return k2, outer, inner

# ── Step 023  unroll_output ──
def _mentions(exprs, rng):
    return any(rng in ranges_in(e) for e in exprs)


def reorder(k, order):
    assert len(order) == len(k.out_ranges)
    assert set(order) == set(k.out_ranges)
    return Kernel(k.name, k.params, list(order), list(k.body), list(k.stores))


def unroll_output(k, rng):
    U = int(rng.src[0].arg)

    for red in k.body:
        for nested in red.body:
            if _mentions(reduce_exprs(nested), rng):
                raise AssertionError("nested reduce mentions rng")

    nid = next_id(k)

    copies = {}
    for red in k.body:
        for acc, init, update in red.accs:
            if _mentions([init, update], rng):
                lst = []
                for _ in range(U):
                    a = UOp(Ops.DEFINE_ACC, acc.dtype, (), nid)
                    nid += 1
                    lst.append(a)
                copies[acc] = lst

    new_body = []
    for red in k.body:
        new_accs = []
        for j in range(U):
            subst = {rng: UOp.const(dtypes.int32, j)}
            for acc in copies:
                subst[acc] = copies[acc][j]
            for acc, init, update in red.accs:
                if acc in copies:
                    a = copies[acc][j]
                    new_accs.append([a, substitute(init, subst), substitute(update, subst)])
        for acc, init, update in red.accs:
            if acc not in copies:
                new_accs.append([acc, init, update])
        new_body.append(Reduce(red.ranges, new_accs, list(red.body)))

    new_stores = []
    for idx_expr, val_expr in k.stores:
        if _mentions([idx_expr, val_expr], rng):
            for j in range(U):
                subst = {rng: UOp.const(dtypes.int32, j)}
                for acc in copies:
                    subst[acc] = copies[acc][j]
                new_stores.append((substitute(idx_expr, subst), substitute(val_expr, subst)))
        else:
            new_stores.append((idx_expr, val_expr))

    new_out = [r for r in k.out_ranges if r is not rng]
    k2 = Kernel(k.name, k.params, new_out, new_body, new_stores)
    return k2.map_exprs(simplify)

# ── Step 024  unroll_reduce ──
def unroll_reduce(k, rng):
    U = int(rng.src[0].arg)

    def find_reduce(red):
        if rng in red.ranges:
            return red
        for nested in red.body:
            found = find_reduce(nested)
            if found is not None:
                return found
        return None

    target = None
    for red in k.body:
        target = find_reduce(red)
        if target is not None:
            break

    assert target is not None, "range not found in any reduce"

    # assert nested bodies of the target do not mention rng
    for nested in target.body:
        for e in reduce_exprs(nested):
            if rng in ranges_in(e):
                raise AssertionError("nested reduce mentions rng")

    # unroll: for each acc, chain U applications of the update
    new_accs = []
    for acc, init, update in target.accs:
        cur = acc
        for j in range(U):
            subst = {rng: UOp.const(dtypes.int32, j), acc: cur}
            cur = substitute(update, subst)
        new_accs.append([acc, init, cur])

    new_target = Reduce(
        [r for r in target.ranges if r is not rng],
        new_accs,
        list(target.body),
    )

    def rebuild(red):
        if red is target:
            return new_target
        return Reduce(list(red.ranges), [[a, i, u] for a, i, u in red.accs],
                      [rebuild(b) for b in red.body])

    new_body = [rebuild(red) for red in k.body]
    k2 = Kernel(k.name, k.params, list(k.out_ranges), new_body, list(k.stores))
    return k2.map_exprs(simplify)

# ── Step 025  optimize_gemm ──
import time


def optimize_gemm(k, tm=4, tn=4, tk=4):
    # k has two output ranges (M, N) and a top-level reduce over K
    m_rng = k.out_ranges[0]
    n_rng = k.out_ranges[1]

    # split M by tm and N by tn
    k, m_outer, m_inner = split_range(k, m_rng, tm)
    k, n_outer, n_inner = split_range(k, n_rng, tn)

    # reorder to [m_outer, n_outer, m_inner, n_inner]
    k = reorder(k, [m_outer, n_outer, m_inner, n_inner])

    # unroll n_inner then m_inner
    k = unroll_output(k, n_inner)
    k = unroll_output(k, m_inner)

    # split the reduce range by tk and unroll the inner part
    if tk > 1:
        red = k.body[0]
        k_rng = red.ranges[0]
        k, k_outer, k_inner = split_range(k, k_rng, tk)
        k = unroll_reduce(k, k_inner)

    return k


def bench_kernel(k, bufs, reps=3):
    import time
    src = render_kernel(k)
    lib = compile_c(src)
    call_kernel(lib, k.name, bufs)  # warm up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        call_kernel(lib, k.name, bufs)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def gflops(n, seconds):
    return 2 * n ** 3 / seconds / 1e9

# ── Step 026  flash_attention_kernel ──
import math


def flash_attention_kernel(N, d, name="flash"):
    out = param("out", dtypes.float32, 0)
    q = param("q", dtypes.float32, 1)
    kk = param("k", dtypes.float32, 2)
    v = param("v", dtypes.float32, 3)

    i = UOp.range(N, 0)
    j = UOp.range(N, 1)
    c = UOp.range(d, 2)

    # nested reduce over d: s = sum_c q[i,c] * k[j,c]
    s_acc = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 3)
    q_load = load(q, i * d + c)
    k_load = load(kk, j * d + c)
    s_update = s_acc + q_load * k_load
    inner = Reduce([c], [[s_acc, UOp.const(dtypes.float32, 0.0), s_update]], [])

    s = s_acc * (1.0 / (d ** 0.5))

    # running max / sum / columns
    m_acc = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 4)
    m_new = m_acc.maximum(s)
    alpha = ((m_acc - m_new) * (1.0 / math.log(2))).exp2()
    p = ((s - m_new) * (1.0 / math.log(2))).exp2()
    m_accs = [m_acc, UOp.const(dtypes.float32, -INF), m_new]

    l_acc = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 5)
    l_accs = [l_acc, UOp.const(dtypes.float32, 0.0), l_acc * alpha + p]

    accs = [m_accs, l_accs]
    col_accs = []
    for col in range(d):
        a = UOp(Ops.DEFINE_ACC, dtypes.float32, (), 6 + col)
        v_load = load(v, j * d + col)
        accs.append([a, UOp.const(dtypes.float32, 0.0), a * alpha + p * v_load])
        col_accs.append(a)

    outer = Reduce([j], accs, [inner])

    stores = [(i * d + col, col_accs[col] * l_acc.recip()) for col in range(d)]
    return Kernel(name, [out, q, kk, v], [i], [outer], stores)

# ── Step 027  to_gpu ──
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

# ── Step 028  render_cuda ──
def render_cuda(k):
    r = CRenderer()

    def emit(line):
        level = len(r.scopes) - 1
        indent = "" if level <= 0 else "    " + "  " * (level - 1)
        r.lines.append(indent + line)

    r.emit = emit

    for p in k.params:
        r.scopes[0][p] = f"data{p.arg[1]}"

    r.push()

    # emit special thread indices, innermost (gidx0 = x) first
    specials = []
    for pos, s in enumerate(k.out_ranges):
        n = len(k.out_ranges)
        axis_from_inner = n - 1 - pos
        d = "xyz"[axis_from_inner]
        i = s.arg[0]
        name = s.arg[1]
        emit(f"int {name} = blockIdx.{d} * blockDim.{d} + threadIdx.{d};")
        specials.append((s, name, int(s.src[0].arg)))

    # guard: all extents
    guard_parts = [f"{name} < {ext}" for (s, name, ext) in specials]
    emit("if (" + " && ".join(guard_parts) + ") {")
    r.push()

    # render reduces (same as render_kernel)
    def render_reduce(red):
        accs = []
        for acc, init, update in red.accs:
            i = acc.arg
            emit(f"{acc.dtype.c_name} acc{i} = {r.expr(init)};")
            r.scopes[-1][acc] = f"acc{i}"
            accs.append((acc, update, i))
        loop_ranges = []
        for rng in red.ranges:
            i = rng.arg
            emit(f"for (int r{i} = 0; r{i} < {rng.src[0].arg}; r{i}++) {{")
            r.push()
            r.scopes[-1][rng] = f"r{i}"
            loop_ranges.append(rng)
        for nested in red.body:
            render_reduce(nested)
        exprs = [r.expr(update) for acc, update, i in accs]
        for (acc, update, i), e in zip(accs, exprs):
            emit(f"acc{i} = {e};")
        for _ in loop_ranges:
            r.pop()
            emit("}")

    for stmt in k.body:
        render_reduce(stmt)

    for idx_expr, val_expr in k.stores:
        emit(f"data0[{r.expr(idx_expr)}] = {r.expr(val_expr)};")

    r.pop()
    emit("}")

    r.pop()

    # signature without restrict
    parts = []
    for p in k.params:
        is_out = p.arg[1] == k.params[0].arg[1]
        const = "" if is_out else "const "
        parts.append(f"{const}{p.dtype.c_name}* data{p.arg[1]}")
    sig = f"__global__ void {k.name}({', '.join(parts)})"

    return "\n".join([sig + " {"] + r.lines + ["}"]) + "\n"


def cuda_source(k, block=(32, 4, 1)):
    kernel_src = render_cuda(k)
    (gx, gy, gz), (bx, by, bz) = launch_dims(k, block)

    parts = []
    for p in k.params:
        is_out = p.arg[1] == k.params[0].arg[1]
        const = "" if is_out else "const "
        parts.append(f"{const}{p.dtype.c_name}* data{p.arg[1]}")
    args = ", ".join(parts)
    call_args = ", ".join(f"data{p.arg[1]}" for p in k.params)

    launcher = (
        f'extern "C" void launch_{k.name}({args}) {{\n'
        f"    dim3 grid({gx}, {gy}, {gz}), block({bx}, {by}, {bz});\n"
        f"    {k.name}<<<grid, block>>>({call_args});\n"
        f"    cudaDeviceSynchronize();\n"
        f"}}"
    )
    return kernel_src + "\n" + launcher + "\n"

# ── Step 029  grad_alu ──
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

# ── Step 030  grad_movement ──
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

# ── Step 031  backward ──
def backward(loss, params):
    # seed gradient of loss with constant 1.0
    grads = {loss.uop: Tensor.const(1.0, loss.dtype)}

    order = loss.uop.toposort()

    for u in reversed(order):
        if u not in grads:
            continue
        g = grads[u]

        if u.op in ALU:
            pairs = grad_alu(u, g)
        elif u.op in MOVEMENT or u.op is Ops.REDUCE_AXIS:
            pairs = grad_movement(u, g)
        else:
            pairs = []

        for src, gt in pairs:
            gt = gt.expand(*shape_of(src))
            if src in grads:
                grads[src] = grads[src] + gt
            else:
                grads[src] = gt

    result = {}
    for p in params:
        result[p] = grads.get(p.uop)
    return result

# ── Step 032  gpt_forward ──
import numpy as np
import math


def gpt_init(V, T, d, rng, scale=0.3):
    def draw(*shape):
        return (rng.standard_normal(shape).astype(np.float32)) * (scale / math.sqrt(shape[0]))
    p = {}
    p['wte'] = draw(V, d)
    p['wpe'] = (rng.standard_normal((T, d)).astype(np.float32) * 0.1) * (scale / math.sqrt(T))
    for name in ('wq', 'wk', 'wv', 'wo'):
        p[name] = draw(d, d)
    p['w1'] = draw(d, 4 * d)
    p['w2'] = draw(4 * d, d)
    p['wout'] = draw(d, V)
    return p


def causal_mask(T):
    m = np.zeros((T, T), dtype=np.float32)
    idx = np.triu_indices(T, k=1)
    m[idx] = -1e9
    return m


def make_batch(B, T, V, rng):
    starts = rng.integers(0, V, size=B)
    seqs = np.stack([(starts + 3 * k) % V for k in range(T + 1)], axis=1)  # (B, T+1)
    def onehot(tokens):
        out = np.zeros((B, T, V), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                out[b, t, tokens[b, t]] = 1.0
        return out
    x = onehot(seqs[:, :T])
    y = onehot(seqs[:, 1:])
    return x, y


def gpt_forward(onehot, mask, p):
    B, T, V = onehot.shape
    d = p['wte'].shape[1]

    x = onehot @ p['wte'] + p['wpe']
    h = x.layernorm()

    q = h @ p['wq']
    k = h @ p['wk']
    v = h @ p['wv']

    scores = (q.reshape(B, T, 1, d) * k.reshape(B, 1, T, d)).sum(3) * (1.0 / math.sqrt(d)) + mask
    pr = scores.softmax(-1)

    y = (pr.reshape(B, T, T, 1) * v.reshape(B, 1, T, d)).sum(2)
    x = x + y @ p['wo']

    x = x + ((x.layernorm() @ p['w1']).relu() @ p['w2'])

    return x.layernorm() @ p['wout']

# ── Step 033  build_train_program ──
import numpy as np


def cross_entropy(logits, targets):
    B, T, V = logits.shape
    return -(logits.logsoftmax(-1) * targets).sum() * (1.0 / (B * T))


def build_train_program(B, T, V, d, lr):
    import numpy as np

    x = Tensor.input('x', (B, T, V))
    y = Tensor.input('y', (B, T, V))
    mask = Tensor.input('mask', (T, T))

    shapes = gpt_init(V, T, d, np.random.default_rng(0))
    params = {k: Tensor.input(k, v.shape) for k, v in shapes.items()}

    logits = gpt_forward(x, mask, params)
    loss = cross_entropy(logits, y)

    grads = backward(loss, list(params.values()))

    outputs = {'loss': loss}
    for name, p in params.items():
        g = grads[p]
        outputs[name + '_new'] = p - lr * g

    return schedule(outputs)

# ── Step 034  train_gpt ──
import time
import numpy as np


def train_gpt(B=4, T=8, V=16, d=16, steps=40, lr=0.5, seed=0):
    import numpy as np
    import time

    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    prog = build_train_program(B, T, V, d, lr)
    cp = Compiled(prog)
    compile_s = time.perf_counter() - t0

    params = gpt_init(V, T, d, rng)
    mask = causal_mask(T)

    losses = []
    step_times = []
    for _ in range(steps):
        x, y = make_batch(B, T, V, rng)
        inputs = {'x': x, 'y': y, 'mask': mask, **params}

        t1 = time.perf_counter()
        out = cp.run(inputs)
        step_times.append(time.perf_counter() - t1)

        losses.append(float(out['loss']))
        params = {k: out[k + '_new'] for k in params}

    return {
        'losses': losses,
        'kernels': len(prog.kernels),
        'unique': cp.unique,
        'compile_s': compile_s,
        'step_ms': (sum(step_times) / len(step_times)) * 1000.0,
        'params': params,
    }

# ── Scaffold (runner) ──
"""Compilers for Machine Learning: a tensor compiler from scratch, measured against tinygrad.

Story: UOps and a rewrite engine simplify symbolic index math; movement ops become
index arithmetic pushed down to loads, so any tensor graph lowers to C loop nests
and a model splits into kernels. Loop transformations turn a naive GEMM into a
register-blocked one, and flash attention is a single fused reduce. The same IR
renders CUDA. Reverse-mode autodiff on the graph trains a tiny GPT entirely through
the compiler, tracking tinygrad's loss curve step for step.
"""
import math
import sys
import time
import numpy as np

sys.setrecursionlimit(100000)


def main() -> None:
    rng = np.random.default_rng(0)

    print("1. Symbolic rewrites")
    r0, r1, r2 = UOp.range(8, 0), UOp.range(4, 1), UOp.range(3, 2)
    flat = (r0 * 4 + r1) * 3 + r2
    e = (flat // 3) % 4
    print(f"   ((r0*4 + r1)*3 + r2) // 3 % 4 has {len(e.toposort())} nodes; after simplification: {len(simplify(e).toposort())} node ({'r1' if simplify(e) is r1 else '?'})")
    a, out = param("a", dtypes.float32, 1), param("out", dtypes.float32, 0)
    s = sink(store(out, 0, (load(a, 0) * 1.0 + 0.0) * 2.0 + (load(a, 1) * 0.0)))
    simp = sink(store(out, 0, simplify(s.src[0].src[1])))
    print(f"   loopless program: {render_sink(s).count(chr(10))} lines of C before rewriting, {render_sink(simp).count(chr(10))} after")

    print("\n2. Compile any model to C")
    xin = Tensor.input("x", (8, 16)); w1 = Tensor.input("w1", (16, 32)); b1 = Tensor.input("b1", (32,)); w2 = Tensor.input("w2", (32, 4))
    q = Tensor.input("wq", (16, 16)); k_ = Tensor.input("wk", (16, 16))
    h = (xin @ w1 + b1).relu()
    att = ((xin @ q) @ (xin @ k_).transpose()).softmax(-1) @ xin
    logits = ((h.layernorm() @ w2) + (att @ Tensor.input("wv", (16, 4)))).softmax(-1)
    prog = schedule({"y": logits})
    inputs = {"x": rng.standard_normal((8, 16)).astype(np.float32), "w1": rng.standard_normal((16, 32)).astype(np.float32) * 0.3,
              "b1": rng.standard_normal(32).astype(np.float32), "w2": rng.standard_normal((32, 4)).astype(np.float32) * 0.3,
              "wq": rng.standard_normal((16, 16)).astype(np.float32) * 0.3, "wk": rng.standard_normal((16, 16)).astype(np.float32) * 0.3,
              "wv": rng.standard_normal((16, 4)).astype(np.float32) * 0.3}
    t0 = time.time(); cp = Compiled(prog); ct = time.time() - t0
    got = cp.run(inputs)["y"]
    ref = eval_tensor(logits.uop, inputs)
    print(f"   MLP + attention block: {len(prog.kernels)} kernels ({cp.unique} unique sources), compiled in {ct:.2f}s, max |C - NumPy| = {float(np.abs(got - ref).max()):.1e}")
    img, wc = Tensor.input("img", (2, 3, 8, 8)), Tensor.input("wc", (4, 3, 3, 3))
    conv = img.conv2d(wc, pad=1).relu()
    kc = lower_kernel(conv.uop, {img.uop: "img", wc.uop: "wc"}, "conv")
    I, Wc = rng.standard_normal((2, 3, 8, 8)).astype(np.float32), rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    got = run_kernel_c(kc, [np.zeros((2, 4, 8, 8), np.float32), I, Wc])
    print(f"   3x3 convolution from movement ops: one kernel, {render_kernel(kc).count('for (')} loops, {len([u for u in kc.body[0].accs[0][2].toposort() if u.op is Ops.LOAD])} loads per step, max err {float(np.abs(got - eval_tensor(conv.uop, {'img': I, 'wc': Wc})).max()):.1e}")

    print("\n3. Fast kernels")
    n = 256
    A, B = Tensor.input("A", (n, n)), Tensor.input("B", (n, n))
    kg = lower_kernel((A @ B).uop, {A.uop: "A", B.uop: "B"}, "gemm")
    X, W = rng.standard_normal((n, n)).astype(np.float32), rng.standard_normal((n, n)).astype(np.float32)
    rows = []
    for label, kk in (("naive triple loop", kg), ("4x4 tile, K by 4", optimize_gemm(kg, 4, 4, 4)), ("4x8 tile, K by 8", optimize_gemm(kg, 4, 8, 8))):
        C = np.zeros((n, n), np.float32)
        t = bench_kernel(kk, [C, X, W])
        rows.append((label, gflops(n, t), float(np.abs(C - X @ W).max())))
    for label, g, err in rows:
        print(f"   {label:20s} {g:7.1f} GFLOP/s  (max err {err:.1e})")
    try:
        from tinygrad import Tensor as TT
        ta, tb = TT(X), TT(W)
        (ta @ tb).realize()
        t0 = time.perf_counter(); (ta @ tb).realize(); tt = time.perf_counter() - t0
        print(f"   {'tinygrad CPU backend':20s} {gflops(n, tt):7.1f} GFLOP/s  (its own codegen and search, same compiler)")
    except Exception as ex:
        print(f"   tinygrad comparison unavailable: {type(ex).__name__}")
    print(f"   speedup of the blocked kernel over the naive loop nest: {rows[2][1] / rows[0][1]:.1f}x")
    N, d = 128, 32
    Q, K_, V = (rng.standard_normal((N, d)).astype(np.float32) for _ in range(3))
    fk = flash_attention_kernel(N, d)
    S = Q @ K_.T / math.sqrt(d); P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); ref = P @ V
    O = np.zeros((N, d), np.float32)
    tf = bench_kernel(fk, [O, Q, K_, V])
    qt, kt, vt = Tensor.input("q", (N, d)), Tensor.input("k", (N, d)), Tensor.input("v", (N, d))
    unf = schedule({"o": ((qt @ kt.transpose()) * (1.0 / math.sqrt(d))).softmax(-1) @ vt})
    cpu = Compiled(unf)
    cpu.run({"q": Q, "k": K_, "v": V})
    t0 = time.perf_counter(); cpu.run({"q": Q, "k": K_, "v": V}); tu = time.perf_counter() - t0
    print(f"   flash attention (N={N}, d={d}): 1 fused kernel, {len(fk.body[0].accs)} accumulators, {1000*tf:.2f} ms, max err {float(np.abs(O - ref).max()):.1e}; unfused schedule: {len(unf.kernels)} kernels, {1000*tu:.2f} ms")

    print("\n4. GPU lowering")
    g = to_gpu(lower_kernel((Tensor.input("A", (64, 48)) @ Tensor.input("B", (48, 32))).uop, {Tensor.input("A", (64, 48)).uop: "A", Tensor.input("B", (48, 32)).uop: "B"}, "gemm"))
    grid, block = launch_dims(g, (32, 4, 1))
    src = render_cuda(g)
    a64, b48 = rng.standard_normal((64, 48)).astype(np.float32), rng.standard_normal((48, 32)).astype(np.float32)
    emu = run_kernel_np(g, [np.zeros((64, 32), np.float32), a64, b48])
    print(f"   64x48 @ 48x32 as a CUDA kernel: grid {grid}, block {block}, {len(src.splitlines())} lines, emulated max err {float(np.abs(emu - a64 @ b48).max()):.1e}")
    print("   " + src.splitlines()[0])
    print("   " + src.splitlines()[3].strip())

    print("\n5. Autodiff and a tiny GPT")
    x, wa, wb, y = Tensor.input("x", (4, 6)), Tensor.input("w1", (6, 8)), Tensor.input("w2", (8, 3)), Tensor.input("y", (4, 3))
    loss = -(((x @ wa).relu().layernorm() @ wb).logsoftmax(-1) * y).sum() * 0.25
    gr = backward(loss, [wa, wb])
    ins = {"x": rng.standard_normal((4, 6)).astype(np.float32), "w1": rng.standard_normal((6, 8)).astype(np.float32) * 0.5, "w2": rng.standard_normal((8, 3)).astype(np.float32) * 0.5}
    Y = np.zeros((4, 3), np.float32); Y[np.arange(4), rng.integers(0, 3, 4)] = 1; ins["y"] = Y
    an = eval_tensor(gr[wa].uop, ins)
    num = np.zeros_like(ins["w1"])
    for i in np.ndindex(num.shape):
        p = dict(ins); u = ins["w1"].copy(); u[i] += 1e-2; p["w1"] = u; lp = float(eval_tensor(loss.uop, p))
        u2 = ins["w1"].copy(); u2[i] -= 1e-2; p["w1"] = u2; lm = float(eval_tensor(loss.uop, p))
        num[i] = (lp - lm) / 2e-2
    print(f"   gradient of an MLP loss vs central differences: max |diff| = {float(np.abs(an - num).max()):.1e}")
    B, T, V, d = 4, 8, 16, 16
    res = train_gpt(B, T, V, d, steps=40, lr=0.5, seed=0)
    print(f"   GPT (V={V}, T={T}, d={d}): {res['kernels']} kernels per step ({res['unique']} unique), compiled in {res['compile_s']:.2f}s, {res['step_ms']:.1f} ms/step")
    print("   loss every 5 steps: " + " ".join(f"{l:.3f}" for l in res["losses"][::5]))
    try:
        from tinygrad import Tensor as TT
        from tinygrad.nn.optim import SGD
        trng = np.random.default_rng(0)
        p0 = gpt_init(V, T, d, trng)
        tp = {k: TT(v.copy()) for k, v in p0.items()}
        for t in tp.values(): t.requires_grad = True
        def tforward(xx, mask):
            Bn, Tn, Vn = xx.shape
            h0 = xx @ tp["wte"] + tp["wpe"]
            hh = h0.layernorm()
            qq, kk, vv = hh @ tp["wq"], hh @ tp["wk"], hh @ tp["wv"]
            sc = (qq.reshape(Bn, Tn, 1, d) * kk.reshape(Bn, 1, Tn, d)).sum(3) * (1.0 / math.sqrt(d)) + mask
            yy = (sc.softmax(-1).reshape(Bn, Tn, Tn, 1) * vv.reshape(Bn, 1, Tn, d)).sum(2)
            h0 = h0 + yy @ tp["wo"]
            h0 = h0 + ((h0.layernorm() @ tp["w1"]).relu() @ tp["w2"])
            return h0.layernorm() @ tp["wout"]
        opt = SGD(list(tp.values()), lr=0.5)
        tl = []
        for _ in range(10):
            xb, yb = make_batch(B, T, V, trng)
            with TT.train():
                tloss = -(tforward(TT(xb), TT(causal_mask(T))).log_softmax(-1) * TT(yb)).sum() * (1.0 / (B * T))
                tl.append(float(tloss.numpy()))
                opt.zero_grad(); tloss.backward(); opt.step()
        print("   tinygrad, same init and batches, first 10 losses: " + " ".join(f"{l:.3f}" for l in tl))
        print("   ours:                                             " + " ".join(f"{l:.3f}" for l in res["losses"][:10]))
        print(f"   max |ours - tinygrad| over 10 steps: {max(abs(a - b) for a, b in zip(tl, res['losses'])):.1e}")
    except Exception as ex:
        print(f"   tinygrad comparison unavailable: {type(ex).__name__}: {ex}")
    print("   The whole training step, forward, loss, backward and SGD, is a program of compiled C kernels produced by this compiler.")


if __name__ == "__main__":
    main()
