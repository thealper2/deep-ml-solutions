def route_topk(x, Wr, bias, k):
    """s = sigmoid(x @ Wr); top-k by s + bias (stable, descending);
    p = raw selected scores normalized per token. Returns (s, idx, p).
    """
    s = 1 / (1 + np.exp(-(x @ Wr)))
    biased = s + bias
    idx = np.argsort(-biased, axis=1, kind='stable')[:, :k]
    p_raw = np.zeros((x.shape[0], k))
    for i in range(x.shape[0]):
        p_raw[i] = s[i, idx[i]]

    p = p_raw / np.sum(p_raw, axis=1, keepdims=True)
    return s, idx, p
