def random_tie_break_argmax(values, candidates, rng):
    """Return one candidate whose value equals max(values), tie-broken uniformly at random."""
    max_value = max(values)
    max_indices = [i for i, v in enumerate(values) if v == max_value]
    idx = rng.integers(0, len(max_indices))
    return candidates[max_indices[idx]]
