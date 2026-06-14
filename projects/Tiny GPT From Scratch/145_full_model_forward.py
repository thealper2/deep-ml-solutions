def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head; return logits and caches."""
    B, T = x_ids.shape
    d_model = model_params['tok_emb'].shape[1]

    tok_emb = model_params['tok_emb'][x_ids]
    pos_emb = model_params['pos_emb'][:T, :]

    x = tok_emb + pos_emb

    emb_cache = {'tok_emb': tok_emb, 'pos_emb': pos_emb, 'x_sum': x}

    blocks_cache = []
    for block_params in model_params['blocks']:
        out = transformer_block_forward(x, block_params)
        x = out['y']
        blocks_cache.append(out['cache'])

    y, ln_cache = final_layernorm_forward(x, model_params['ln_f']['gamma'], model_params['ln_f']['beta'])
    logits = y @ model_params['lm_head']['w_lm'] + model_params['lm_head']['b_lm']

    lm_cache = {'w_lm': model_params['lm_head']['w_lm'], 'b_lm': model_params['lm_head']['b_lm']}

    caches = {
        'emb': emb_cache,
        'blocks': blocks_cache,
        'ln_f': ln_cache,
        'lm_head': lm_cache,
    }

    return logits, caches
