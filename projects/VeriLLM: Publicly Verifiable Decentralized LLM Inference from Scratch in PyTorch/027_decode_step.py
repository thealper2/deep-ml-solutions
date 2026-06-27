def decode_step(prev_token_id, kv_caches, next_pos, model_params):
    token_embeds = embed_tokens(np.array([prev_token_id]), model_params['token_embedding'])
    
    x = add_positional_embeddings(token_embeds, model_params['pos_embedding'], start_pos=next_pos)
    
    new_kv_caches = []
    for block_idx, block_params in enumerate(model_params['blocks']):
        kv_cache = kv_caches[block_idx]
        x, kv_cache = transformer_block(x, block_params, kv_cache, query_offset=next_pos)
        new_kv_caches.append(kv_cache)
    
    hidden = layer_norm_apply(x, model_params['ln_f'])
    
    lm_head = model_params['lm_head']
    logits = linear_projection(hidden, lm_head['W'], lm_head.get('b', None))
    logits = logits.flatten()
    
    next_token = greedy_next_token(logits)
    
    return {
        'next_token': next_token,
        'logits': logits,
        'kv_caches': new_kv_caches,
        'next_pos': next_pos + 1
    }
