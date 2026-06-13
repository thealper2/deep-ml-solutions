def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    logits = w[ids]

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    B = len(ids)
    one_hot = np.zeros((B, w.shape[0]))
    one_hot[np.arange(B), targets] = 1.0
    loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-15), axis=1))

    dlogits = (probs - one_hot) / B

    vocab_size = w.shape[0]
    dW = np.zeros((vocab_size, vocab_size))
    np.add.at(dW, ids, dlogits)

    new_w = w - learning_rate * dW
    return {'w': new_w, 'loss': float(loss)}
