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
        self.out_ranges = tuple(out_ranges)
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