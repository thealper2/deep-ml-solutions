import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    dW = np.zeros((vocab_size, dlogits.shape[1]))
    np.add.at(dW, ids, dlogits)
    return dW
