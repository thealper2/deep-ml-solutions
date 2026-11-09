import numpy as np


def _as_f32(a):
    return np.asarray(a, dtype=np.float32)


def _eval(node, bufs, cache):
    key = node
    if key in cache:
        return cache[key]

    op = node[0]

    if op == "buf":
        val = _as_f32(bufs[node[1]])
    elif op == "const":
        val = np.float32(node[1])
    elif op in ("add", "mul", "max"):
        a = _eval(node[1], bufs, cache)
        b = _eval(node[2], bufs, cache)
        if op == "add":
            val = _as_f32(a + b)
        elif op == "mul":
            val = _as_f32(a * b)
        else:
            val = _as_f32(np.maximum(a, b))
    elif op in ("neg", "exp", "recip"):
        a = _eval(node[1], bufs, cache)
        if op == "neg":
            val = _as_f32(-a)
        elif op == "exp":
            val = _as_f32(np.exp(a))
        else:
            val = _as_f32(np.float32(1.0) / a)
    elif op == "reshape":
        a = _eval(node[1], bufs, cache)
        val = _as_f32(np.reshape(a, node[2]))
    elif op == "expand":
        a = _eval(node[1], bufs, cache)
        val = _as_f32(np.broadcast_to(a, node[2]))
    elif op == "permute":
        a = _eval(node[1], bufs, cache)
        val = _as_f32(np.transpose(a, node[2]))
    elif op == "flip":
        a = _eval(node[1], bufs, cache)
        val = _as_f32(np.flip(a, node[2]))
    elif op == "pad":
        a = _eval(node[1], bufs, cache)
        val = _as_f32(np.pad(a, node[2]))
    elif op == "shrink":
        a = _eval(node[1], bufs, cache)
        slices = tuple(slice(int(s), int(e)) for s, e in node[2])
        val = _as_f32(a[slices])
    elif op == "sum":
        a = _eval(node[1], bufs, cache)
        val = _as_f32(np.sum(a, axis=node[2], keepdims=True, dtype=np.float32))
    elif op == "amax":
        a = _eval(node[1], bufs, cache)
        val = _as_f32(np.max(a, axis=node[2], keepdims=True))
    else:
        raise ValueError(f"unknown op {op!r}")

    cache[key] = val
    return val


def evaluate(node, bufs, cache=None):
    if cache is None:
        cache = {}
    return _eval(node, bufs, cache)


def eval_count(node, bufs):
    cache = {}
    _eval(node, bufs, cache)
    return len(cache)