def log_softmax(logits, axis=-1):
    max_logits = np.max(logits, axis=axis, keepdims=True)
    log_sum_exp = max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=axis, keepdims=True))
    return logits - log_sum_exp
