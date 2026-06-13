def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    d_token_emb = d_out
    d_pos_emb = sum_axis0(np.sum(d_out, axis=0))
    d_pos_emb = np.sum(d_out, axis=0)
    return {'d_token_emb': d_token_emb, 'd_pos_emb': d_pos_emb}
