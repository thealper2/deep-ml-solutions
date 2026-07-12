def gather_token_logprobs(log_probs, token_ids):
    B, T, V = log_probs.shape
    result = np.zeros((B, T))
    for b in range(B):
        for t in range(T):
            result[b, t] = log_probs[b, t, token_ids[b, t]]

    return result
