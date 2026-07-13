def policy_sequence_logprob(params, token_ids, mask):
    logits = policy_token_logits(params, token_ids)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    log_probs = logits - max_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    B, T, V = logits.shape
    token_logprobs = np.zeros((B, T))
    for b in range(B):
        for t in range(T):
            token_logprobs[b, t] = log_probs[b, t, token_ids[b, t]]

    return np.sum(token_logprobs * mask, axis=1)
