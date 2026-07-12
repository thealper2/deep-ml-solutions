def softmax(logits, axis=-1):
    max_logits = np.max(logits, axis=axis, keepdims=True)
    exp_values = np.exp(logits - max_logits)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
