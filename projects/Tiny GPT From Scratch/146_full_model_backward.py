def full_model_backward(d_logits, caches, model_params):
    """Propagate d_logits back through LM head, final LN, blocks, and embeddings.

    Args:
        d_logits: (B, T, V) gradient w.r.t. the model output
        caches: nested dict from full_model_forward with keys
                'emb', 'blocks', 'ln_f', 'lm_head'
        model_params: nested dict matching the forward's parameter tree

    Returns:
        grads: nested dict mirroring model_params with keys
               'tok_emb', 'pos_emb', 'blocks', 'ln_f': {'gamma', 'beta'},
               'lm_head': {'w_lm', 'b_lm'}
    """
    lm_cache = caches['lm_head']
    x_lm = lm_cache['x']
    w_lm = lm_cache['w_lm']
    
    d_x_lm = d_logits @ w_lm.T
    d_w_lm = (x_lm.reshape(-1, x_lm.shape[-1]).T @ d_logits.reshape(-1, d_logits.shape[-1]))
    d_b_lm = np.sum(d_logits, axis=(0, 1))
    
    ln_cache = caches['ln_f']
    x_ln = ln_cache['x']
    mean = ln_cache['mean']
    var = ln_cache['var']
    gamma = ln_cache['gamma']
    eps = ln_cache.get('eps', 1e-5)
    
    D = x_ln.shape[-1]
    std = np.sqrt(var + eps)
    x_hat = ln_cache['x_hat']
    
    d_gamma = np.sum(d_x_lm * x_hat, axis=(0, 1))
    d_beta = np.sum(d_x_lm, axis=(0, 1))
    
    d_x_hat = d_x_lm * gamma
    d_var = np.sum(d_x_hat * (x_ln - mean) * -0.5 * (var + eps) ** -1.5, axis=-1, keepdims=True)
    d_mean = np.sum(d_x_hat * -1 / std, axis=-1, keepdims=True) + d_var * np.mean(-2 * (x_ln - mean), axis=-1, keepdims=True)
    d_x_ln = d_x_hat / std + d_var * 2 * (x_ln - mean) / D + d_mean / D
    
    blocks_caches = caches['blocks']
    blocks_params = model_params['blocks']
    d_x_blocks, blocks_grads = backward_through_all_blocks(d_x_ln, blocks_caches, blocks_params)
    
    emb_cache = caches['emb']
    d_emb = d_x_blocks
    
    tok_cache = emb_cache['tok_cache']
    token_ids = tok_cache['token_ids']
    
    d_tok_emb = np.zeros_like(model_params['tok_emb'])
    d_emb_flat = d_emb.reshape(-1, d_emb.shape[-1])
    token_ids_flat = token_ids.reshape(-1)
    np.add.at(d_tok_emb, token_ids_flat, d_emb_flat)
    
    T_seq = emb_cache['seq_len']
    d_pos_emb = np.zeros_like(model_params['pos_emb'])
    d_pos_emb[:T_seq, :] = np.sum(d_emb, axis=0)
    
    grads = {
        'tok_emb': d_tok_emb,
        'pos_emb': d_pos_emb,
        'blocks': blocks_grads,
        'ln_f': {
            'gamma': d_gamma,
            'beta': d_beta
        },
        'lm_head': {
            'w_lm': d_w_lm,
            'b_lm': d_b_lm
        }
    }
    
    return grads
