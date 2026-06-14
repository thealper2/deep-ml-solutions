def softmax_to_probs(logits):
    """Convert (1, V) logits into a row-wise probability distribution."""
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs
