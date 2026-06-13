def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    return np.array([prob[target] for prob, target in zip(probs, targets)])
