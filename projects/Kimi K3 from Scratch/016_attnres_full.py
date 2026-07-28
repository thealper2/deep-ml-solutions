def attnres_full(pseudo_q, sources):
    """h[t] = sum_i attnres_weights(...)[i, t] * sources[i][t] (raw values).

    Returns (T, d).
    """
    weights = attnres_weights(pseudo_q, sources)
    n, T = weights.shape
    d = sources[0].shape[1]

    h = np.zeros((T, d))
    for i in range(n):
        h += weights[i, :, None] * sources[i]

    return h
