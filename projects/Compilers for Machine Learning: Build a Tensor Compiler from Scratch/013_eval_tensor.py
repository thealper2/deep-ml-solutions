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