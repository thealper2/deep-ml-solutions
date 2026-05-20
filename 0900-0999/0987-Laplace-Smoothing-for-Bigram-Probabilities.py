import numpy as np

def smooth_bigram_probs(N, k):
    """
    N: 2D list or array of bigram counts (V x V)
    k: float, add-k smoothing constant (k >= 0)
    Returns: 2D list of smoothed bigram probabilities (V x V).
    """
    N = np.array(N)
    added = N + k
    row_sums = np.sum(added, axis=1)
    divided = added / row_sums[:, np.newaxis]
    return divided.tolist()