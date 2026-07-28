def topk_cutoffs(s, bias, k):
    """Top-(k+1) on s + bias: first k -> routes (m, k); (k+1)-th biased score
    -> cutoffs (m,). Returns (routes, cutoffs).
    """
    biased = s + bias
    idx_sorted = np.argsort(-biased, axis=1, kind='stable')
    routes = idx_sorted[:, :k]
    cutoffs = biased[np.arange(biased.shape[0]), idx_sorted[:, k]]
    return routes, cutoffs
