def short_conv(x, w):
    """Causal depthwise conv: y[t,c] = sum_j w[j,c] * x[t-(K-1)+j, c].

    x: (T, d) sequence.  w: (K, d) per-channel kernel, w[K-1] = current token.
    Positions before the sequence start count as zeros.
    """
    T, d = x.shape
    K = w.shape[0]
    y = np.zeros_like(x)

    for j in range(K):
        shift = K - 1 - j
        y[shift:] += w[j] * x[:-shift] if shift > 0 else w[j] * x

    return y
