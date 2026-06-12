import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    counts = allocate_count_matrix(vocab_size)
    current = data[:-1]
    next_tokens = data[1:]
    np.add.at(counts, (current, next_tokens), 1)
    return counts
