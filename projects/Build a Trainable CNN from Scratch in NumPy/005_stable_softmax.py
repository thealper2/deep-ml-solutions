def stable_softmax(logits):
    shifted_exp = exp_shifted(logits)
    return shifted_exp / np.sum(shifted_exp, axis=1, keepdims=True)
