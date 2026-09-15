import numpy as np


def magnitude_prune(W, sparsity):
    W = np.asarray(W, dtype=float)
    n_prune = int(np.floor(sparsity * W.size))
    flat_idx = np.argsort(np.abs(W).ravel(), kind='stable')[:n_prune]
    mask = np.ones(W.size, dtype=bool)
    mask[flat_idx] = False
    mask = mask.reshape(W.shape)
    pruned = np.where(mask, W, 0.0)
    return pruned, mask

def global_magnitude_prune(weights, sparsity):
    sizes = [w.size for w in weights]
    shapes = [w.shape for w in weights]
    mags = np.concatenate([np.abs(w).ravel() for w in weights])
    total = mags.size
    n_prune = int(np.floor(sparsity * total))
    flat_idx = np.argsort(mags, kind='stable')[:n_prune]
    mask_flat = np.ones(total, dtype=bool)
    mask_flat[flat_idx] = False

    result = []
    offset = 0
    for w, s, shp in zip(weights, sizes, shapes):
        m = mask_flat[offset:offset+s].reshape(shp)
        pruned = np.where(m, np.asarray(w, dtype=float), 0.0)
        result.append((pruned, m))
        offset += s
    
    return result

def actual_sparsity(mask):
    mask = np.asarray(mask)
    return float(1.0 - mask.mean())