from enum import Enum, auto


class Ops(Enum):
    CONST = auto(); PARAM = auto(); ADD = auto(); MUL = auto(); MAX = auto(); CMPLT = auto(); AND = auto(); IDIV = auto(); MOD = auto()
    WHERE = auto(); RECIP = auto(); EXP2 = auto(); SQRT = auto(); RANGE = auto(); SPECIAL = auto(); INDEX = auto(); LOAD = auto()
    DEFINE_ACC = auto(); LOCAL = auto()


class DType:
    def __init__(self, name, c): self.name, self.c = name, c
    def __repr__(self): return f"dtypes.{self.name}"


class dtypes:
    int32 = DType("int32", "int"); float32 = DType("float32", "float"); bool = DType("bool", "int"); half = DType("half", "__half")


def _as_uop(x, dtype=None):
    if isinstance(x, UOp): return x
    if dtype is None: dtype = dtypes.bool if isinstance(x, bool) else dtypes.int32 if isinstance(x, int) else dtypes.float32
    return UOp.const(dtype, x)


class UOp:
    __slots__ = ("op", "dtype", "src", "arg")
    _cache = {}

    def __new__(cls, op, dtype=None, src=(), arg=None):
        src = tuple(_as_uop(s) for s in src)
        key = (op, dtype, src, (type(arg), arg))
        if (ret := cls._cache.get(key)) is not None: return ret
        ret = super().__new__(cls)
        ret.op, ret.dtype, ret.src, ret.arg = op, dtype, src, arg
        cls._cache[key] = ret
        return ret

    def __repr__(self): return f"UOp({self.op.name}, {self.dtype}, {self.src!r}, arg={self.arg!r})"

    @staticmethod
    def const(dtype, v):
        return UOp(Ops.CONST, dtype, (), int(v) if dtype in (dtypes.int32, dtypes.bool) else float(v))

    def _wrap(self, x): return _as_uop(x, self.dtype)

    def alu(self, op, *src):
        srcs = (self,) + tuple(self._wrap(s) for s in src)
        return UOp(op, dtypes.bool if op in (Ops.CMPLT, Ops.AND) else self.dtype, srcs)

    def __add__(self, x): return self.alu(Ops.ADD, x)
    def __radd__(self, x): return self._wrap(x).alu(Ops.ADD, self)
    def __mul__(self, x): return self.alu(Ops.MUL, x)
    def __rmul__(self, x): return self._wrap(x).alu(Ops.MUL, self)
    def __floordiv__(self, x): return self.alu(Ops.IDIV, x)
    def __rfloordiv__(self, x): return self._wrap(x).alu(Ops.IDIV, self)
    def __mod__(self, x): return self.alu(Ops.MOD, x)
    def __rmod__(self, x): return self._wrap(x).alu(Ops.MOD, self)
    def __lt__(self, x): return self.alu(Ops.CMPLT, x)
    def __and__(self, x): return self.alu(Ops.AND, x)
    def __rand__(self, x): return self._wrap(x).alu(Ops.AND, self)
    def maximum(self, x): return self.alu(Ops.MAX, x)

    def where(self, a, b):
        dt = a.dtype if isinstance(a, UOp) else b.dtype if isinstance(b, UOp) else _as_uop(a).dtype
        return UOp(Ops.WHERE, dt, (self, _as_uop(a, dt), _as_uop(b, dt)))

    def toposort(self):
        visited, order, stack = set(), [], [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if node in visited: continue
            visited.add(node)
            stack.append((node, True))
            for s in reversed(node.src):
                if s not in visited: stack.append((s, False))
        return order


def param(name, dtype, i): return UOp(Ops.PARAM, dtype, (), (name, i))
def rng_(n, i): return UOp(Ops.RANGE, dtypes.int32, (UOp.const(dtypes.int32, n),), i)
def special(name): return UOp(Ops.SPECIAL, dtypes.int32, (), name)
def load(p, idx): return UOp(Ops.LOAD, p.dtype, (UOp(Ops.INDEX, p.dtype, (p, _as_uop(idx, dtypes.int32))),))
def acc_(dtype, i): return UOp(Ops.DEFINE_ACC, dtype, (), i)
def local(name, dtype, size): return UOp(Ops.LOCAL, dtype, (), (name, size))