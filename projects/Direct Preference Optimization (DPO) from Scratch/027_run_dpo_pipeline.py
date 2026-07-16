def run_dpo_pipeline(vocab_size, d_model, prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta, learning_rate, num_steps, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    params = init_policy_params(vocab_size, d_model, rng=rng)
    pairs = build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    ref_logprobs = freeze_reference_logprobs(params, pairs)
    trained_params, history = train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng)
    eval_metrics = evaluate_dpo(trained_params, pairs, ref_logprobs, beta)

    return {
        'params': trained_params,
        'history': history,
        'eval_metrics': eval_metrics,
    }
