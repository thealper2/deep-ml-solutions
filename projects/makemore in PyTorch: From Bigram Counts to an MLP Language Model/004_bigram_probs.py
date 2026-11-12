def bigram_probs(N, smoothing=1):
    """Row-normalized float probabilities after adding `smoothing` to every count."""
    N = N.float() + smoothing
    return N / N.sum(dim=1, keepdim=True)