def greedy_next_token(logits):
    if logits.ndim == 1:
        return np.argmax(logits, axis=-1)
    else:
        max_val = np.argmax(logits[-1], axis=-1)
        return max_val
