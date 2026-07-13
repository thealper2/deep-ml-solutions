def sequence_logprob_grad(params, token_ids, mask):
    embed, W_out, b_out = params['embed'], params['W_out'], params['b_out']
    token_ids = np.asarray(token_ids)
    mask = np.asarray(mask, dtype=float)

    h = embed[token_ids]
    logits = h @ W_out + b_out

    p = softmax(logits)
    one_hot = np.zeros_like(p)
    B, T = token_ids.shape
    bi, ti = np.indices((B, T))
    one_hot[bi, ti, token_ids] = 1.0

    dlogits = (one_hot - p) * mask[..., None]

    db_out = dlogits.sum(axis=(0, 1))
    dW_out = np.einsum('btd,btv->dv', h, dlogits)
    dh = dlogits @ W_out.T

    dembed = np.zeros_like(embed)
    np.add.at(dembed, token_ids, dh)

    return {'embed': dembed, 'W_out': dW_out, 'b_out': db_out}
