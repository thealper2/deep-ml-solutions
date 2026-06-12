def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    probs = p_matrix[current_id]
    return rng.choice(len(probs), p=probs)
