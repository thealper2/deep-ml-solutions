def generation_loop_for_n_steps(params, prompt_ids, n_new_tokens, block_size, temperature, top_k, rng):
    """Iteratively generate n_new_tokens by repeatedly forwarding the cropped context."""
    context = prompt_ids.copy()

    for _ in range(n_new_tokens):
        if context.shape[1] > block_size:
            context = crop_context_to_block_size(context, block_size)

        logits = forward_to_get_logits(params, context)
        last_logits = take_last_position_logits(logits)
        last_logits = last_logits / temperature
        filtered_logits = top_k_filter(last_logits, top_k)
        probs = softmax_to_probs(filtered_logits)
        next_token = sample_one_token(probs, rng)
        context = append_token_to_sequence(context, next_token)

    return context
