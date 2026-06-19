def softmax_cross_entropy_backward(logits, y):
    probs = stable_softmax(logits)
    N, C = logits.shape
    y_one_hot = one_hot(y, C)
    dlogits = (probs - y_one_hot) / N
    return dlogits
