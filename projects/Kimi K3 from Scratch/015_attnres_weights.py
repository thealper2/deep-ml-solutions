def attnres_weights(pseudo_q, sources):
    """Softmax over depth: w[i, t] prop. to exp(pseudo_q . RMSNorm(sources[i][t])).

    sources: list of n (T, d) arrays.  Returns (n, T); columns sum to 1.
    """
    n = len(sources)
    T = sources[0].shape[0]

    logits = np.zeros((n, T))
    for i, src in enumerate(sources):
        rms = np.sqrt(np.mean(src ** 2, axis=1, keepdims=True) + 1e-6)
        src_norm = src / rms
        logits[i] = src_norm @ pseudo_q

    max_logits = np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    weights = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    return weights
