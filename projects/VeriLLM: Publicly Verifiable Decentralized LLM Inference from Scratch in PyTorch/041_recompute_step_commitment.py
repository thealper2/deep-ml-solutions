def recompute_step_commitment(reexec_state, prior_kv_cache):
    step_state = {
        'step_index': reexec_state.get('step_index', 0),
        'input_token': reexec_state.get('input_token', 0),
        'next_token': reexec_state.get('token', reexec_state.get('next_token', 0)),
        'logits': reexec_state.get('logits', np.array([])),
        'kv_caches': reexec_state.get('kv_cache_after', reexec_state.get('kv_caches', [])),
        'next_pos': reexec_state.get('next_pos', 0)
    }
    
    return commit_decode_step(step_state)
