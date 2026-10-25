def run_kernel_np(k, bufs):
    def rsize(r):
        return int(r.src[0].arg) if r.op is Ops.RANGE else 1

    shape = tuple(rsize(r) for r in k.out_ranges)

    if shape:
        grid = np.indices(shape)
    else:
        grid = np.empty((0,), dtype=np.int64)

    env = {}
    for i, r in enumerate(k.out_ranges):
        if r.op is Ops.RANGE:
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