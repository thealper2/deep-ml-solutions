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