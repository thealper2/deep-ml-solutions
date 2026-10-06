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