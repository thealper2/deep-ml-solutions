import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    V = w.shape[0]
    B = len(ids)

    one_hot = np.zeros((B, V))
    one_hot[np.arange(B), ids] = 1.0
    onehot_result = one_hot @ w

    index_result = w[ids]

    return {
        'onehot_result': onehot_result,
        'index_result': index_result,
    }
