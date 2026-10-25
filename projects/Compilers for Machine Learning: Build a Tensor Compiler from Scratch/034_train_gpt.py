import time
import numpy as np


def train_gpt(B=4, T=8, V=16, d=16, steps=40, lr=0.5, seed=0):
    import numpy as np
    import time

    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    prog = build_train_program(B, T, V, d, lr)
    cp = Compiled(prog)
    compile_s = time.perf_counter() - t0

    params = gpt_init(V, T, d, rng)
    mask = causal_mask(T)

    losses = []
    step_times = []
    for _ in range(steps):
        x, y = make_batch(B, T, V, rng)
        inputs = {'x': x, 'y': y, 'mask': mask, **params}

        t1 = time.perf_counter()
        out = cp.run(inputs)
        step_times.append(time.perf_counter() - t1)

        losses.append(float(out['loss']))
        params = {k: out[k + '_new'] for k in params}

    return {
        'losses': losses,
        'kernels': len(prog.kernels),
        'unique': cp.unique,
        'compile_s': compile_s,
        'step_ms': (sum(step_times) / len(step_times)) * 1000.0,
        'params': params,
    }