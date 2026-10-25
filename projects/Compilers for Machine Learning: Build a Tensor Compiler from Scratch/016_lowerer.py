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
                out_ranges.append(UOp.const(dtypes.int32, 0))
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