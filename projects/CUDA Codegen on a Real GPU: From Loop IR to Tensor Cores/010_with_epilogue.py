def with_epilogue(k, fn, extra_params=()):
    return Kernel(k.name, list(k.params) + list(extra_params), k.specials, k.body,
                  [(idx, fn(val, idx), cond) for idx, val, cond in k.stores],
                  k.block, k.grid, k.locals_, k.tiles)


def bias_relu(bias, N):
    return lambda v, idx: (v + load(bias, idx % N)).maximum(0.0)