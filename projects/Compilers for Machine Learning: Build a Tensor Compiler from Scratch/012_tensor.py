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