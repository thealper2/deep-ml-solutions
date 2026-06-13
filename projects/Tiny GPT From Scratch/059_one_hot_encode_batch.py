import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    return np.eye(vocab_size)[ids]
