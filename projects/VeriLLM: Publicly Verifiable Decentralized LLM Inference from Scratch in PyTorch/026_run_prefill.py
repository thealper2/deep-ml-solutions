def run_prefill(prompt_ids, model_params):
    """Run prefill over the prompt tokens and build the initial KV cache per layer."""
    token_embeds = embed_tokens(prompt_ids, model_params['token_embedding'])
    x = add_positional_embeddings(token_embeds, model_params['pos_embedding'], start_pos=0)
    kv_caches = []
    for block_idx, block_params in enumerate(model_params['blocks']):
        kv_cache = {'k': None, 'v': None}
        x, kv_cache = transformer_block(x, block_params, kv_cache, query_offset=0)
        kv_caches.append(kv_cache)

    hidden = layer_norm_apply(x, model_params['ln_f'])
    return {
        'hidden': hidden,
        'kv_caches': kv_caches,
        'next_pos': len(prompt_ids),
    }
