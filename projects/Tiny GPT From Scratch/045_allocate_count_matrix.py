import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    return np.zeros((vocab_size, vocab_size), dtype=np.int64)
