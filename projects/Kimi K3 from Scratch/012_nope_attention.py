def nope_attention(x, Wq, Wc, Wk_up, Wv_up, n_heads):
    """Causal multi-head attention over MLA-reconstructed K,V - no positions.

    Q = (x @ Wq).reshape(T, H, dh); per head softmax(QK^T/sqrt(dh)) V with a
    causal mask; concatenate heads -> (T, H*dh).
    """
    T, d = x.shape
    dh = Wq.shape[1] // n_heads
    Q = (x @ Wq).reshape(T, n_heads, dh)
    _, K, V = mla_compress_reconstruct(x, Wc, Wk_up, Wv_up, n_heads)
    out = np.zeros((T, n_heads, dh))

    for h in range(n_heads):
        Q_h = Q[:, h, :]
        K_h = K[:, h, :]
        V_h = V[:, h, :]

        scores = (Q_h @ K_h.T) / np.sqrt(dh)

        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores[mask] = -1e9

        scores_stable = scores - np.max(scores, axis=1, keepdims=True)
        attn = np.exp(scores_stable)
        attn = attn / np.sum(attn, axis=1, keepdims=True)

        out[:, h, :] = attn @ V_h

    return out.reshape(T, -1)
