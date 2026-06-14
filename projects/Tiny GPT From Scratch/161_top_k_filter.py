def top_k_filter(logits, k):
    """Return logits with all but the top-k entries per row set to -inf."""
    result = np.full_like(logits, -np.inf)
    top_k_indices = np.argsort(logits, axis=-1)[:, -k:]
    for i in range(logits.shape[0]):
        result[i, top_k_indices[i]] = logits[i, top_k_indices[i]]
    
    return result
