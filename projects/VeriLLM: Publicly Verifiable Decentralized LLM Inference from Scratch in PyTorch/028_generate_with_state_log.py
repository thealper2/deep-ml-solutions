def generate_with_state_log(prompt_ids, model_params, num_new_tokens):
    """Run prefill, then decode num_new_tokens tokens, logging each step's state."""
    prefill_result = run_prefill(prompt_ids, model_params)
    kv_caches = prefill_result['kv_caches']
    next_pos = prefill_result['next_pos']
    hidden = prefill_result['hidden']

    generated_tokens = []
    step_states = []

    if num_new_tokens > 0:
        last_hidden = np.asarray(hidden)[-1]
        logits = lm_head_logits(last_hidden, model_params['lm_head'])
        logits = np.asarray(logits).flatten()
        first_token = greedy_next_token(logits)

        step_states.append({
            'next_token': first_token,
            'logits': logits,
            'kv_caches': kv_caches,
            'next_pos': next_pos,
        })
        generated_tokens.append(first_token)
        prev_token_id = first_token

        for _ in range(num_new_tokens - 1):
            result = decode_step(prev_token_id, kv_caches, next_pos, model_params)

            kv_caches = result['kv_caches']
            next_pos = result['next_pos']

            step_states.append({
                'next_token': result['next_token'],
                'logits': result['logits'],
                'kv_caches': kv_caches,
                'next_pos': next_pos,
            })
            generated_tokens.append(result['next_token'])
            prev_token_id = result['next_token']

    return {
        'generated_tokens': generated_tokens,
        'step_states': step_states,
    }
