import numpy as np
import math


def gpt_init(V, T, d, rng, scale=0.3):
    def draw(*shape):
        return (rng.standard_normal(shape).astype(np.float32)) * (scale / math.sqrt(shape[0]))
    p = {}
    p['wte'] = draw(V, d)
    p['wpe'] = (rng.standard_normal((T, d)).astype(np.float32) * 0.1) * (scale / math.sqrt(T))
    for name in ('wq', 'wk', 'wv', 'wo'):
        p[name] = draw(d, d)
    p['w1'] = draw(d, 4 * d)
    p['w2'] = draw(4 * d, d)
    p['wout'] = draw(d, V)
    return p


def causal_mask(T):
    m = np.zeros((T, T), dtype=np.float32)
    idx = np.triu_indices(T, k=1)
    m[idx] = -1e9
    return m


def make_batch(B, T, V, rng):
    starts = rng.integers(0, V, size=B)
    seqs = np.stack([(starts + 3 * k) % V for k in range(T + 1)], axis=1)  # (B, T+1)
    def onehot(tokens):
        out = np.zeros((B, T, V), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                out[b, t, tokens[b, t]] = 1.0
        return out
    x = onehot(seqs[:, :T])
    y = onehot(seqs[:, 1:])
    return x, y


def gpt_forward(onehot, mask, p):
    B, T, V = onehot.shape
    d = p['wte'].shape[1]

    x = onehot @ p['wte'] + p['wpe']
    h = x.layernorm()

    q = h @ p['wq']
    k = h @ p['wk']
    v = h @ p['wv']

    scores = (q.reshape(B, T, 1, d) * k.reshape(B, 1, T, d)).sum(3) * (1.0 / math.sqrt(d)) + mask
    pr = scores.softmax(-1)

    y = (pr.reshape(B, T, T, 1) * v.reshape(B, 1, T, d)).sum(2)
    x = x + y @ p['wo']

    x = x + ((x.layernorm() @ p['w1']).relu() @ p['w2'])

    return x.layernorm() @ p['wout']