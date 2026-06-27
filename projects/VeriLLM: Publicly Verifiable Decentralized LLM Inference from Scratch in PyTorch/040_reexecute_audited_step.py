def reexecute_audited_step(model_params, prior_kv_cache, prior_token):
    next_pos = prior_kv_cache[0]['k'].shape[0] if prior_kv_cache else 0

    result = decode_step(prior_token, prior_kv_cache, next_pos, model_params)

    return {
        'hidden': result.get('hidden'),
        'logits': result['logits'],
        'token': int(result['next_token']),
        'kv_cache_after': result['kv_caches'],
    }
