def masked_sequence_logprob(token_logprobs, mask):
    return np.sum(token_logprobs * mask, axis=1)
