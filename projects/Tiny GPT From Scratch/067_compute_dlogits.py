def compute_dlogits(probs, targets):
    B, V = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(B), targets] = 1.0
    return (probs - one_hot) / B
