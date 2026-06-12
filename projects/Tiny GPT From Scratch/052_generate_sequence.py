def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    sequence = np.zeros(length, dtype=np.int64)
    sequence[0] = start_id
    
    for i in range(1, length):
        sequence[i] = sample_next_token(p_matrix, sequence[i - 1], rng)
    
    return sequence
