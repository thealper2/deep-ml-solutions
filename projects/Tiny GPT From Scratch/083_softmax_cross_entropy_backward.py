def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    B, V = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(B), targets] = 1.0
    return (probs - one_hot) / B
