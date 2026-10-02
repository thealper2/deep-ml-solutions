import numpy as np


def block_diagonal(mats):
    mats = [np.asarray(m, dtype=float) for m in mats]
    if not mats:
        return np.zeros((0, 0), dtype=float)
    total = sum(m.shape[0] for m in mats)
    out = np.zeros((total, total), dtype=float)
    offset = 0
    for m in mats:
        n = m.shape[0]
        out[offset:offset + n, offset:offset + n] = m
        offset += n
    return out


def batch_graphs(adjs, feats):
    adjs = [np.asarray(a) for a in adjs]
    feats = [np.asarray(f, dtype=float) for f in feats]
    sizes = [a.shape[0] for a in adjs]

    A = block_diagonal(adjs)
    X = np.vstack(feats) if feats else np.zeros((0, 0), dtype=float)
    batch = np.concatenate([np.full(n, g, dtype=np.int64) for g, n in enumerate(sizes)])
    return A, X, batch


def readout(H, batch, mode="sum"):
    H = np.asarray(H, dtype=float)
    batch = np.asarray(batch)
    G = int(batch.max()) + 1
    d = H.shape[1]
    out = np.zeros((G, d), dtype=float)

    if mode == "sum":
        for g in range(G):
            out[g] = H[batch == g].sum(axis=0)
    elif mode == "mean":
        for g in range(G):
            mask = batch == g
            if mask.any():
                out[g] = H[mask].mean(axis=0)
    elif mode == "max":
        for g in range(G):
            mask = batch == g
            if mask.any():
                out[g] = H[mask].max(axis=0)
    else:
        raise ValueError("mode must be 'sum', 'mean', or 'max'")

    return out


def unbatch(H, batch):
    H = np.asarray(H)
    batch = np.asarray(batch)
    G = int(batch.max()) + 1
    return [H[batch == g] for g in range(G)]