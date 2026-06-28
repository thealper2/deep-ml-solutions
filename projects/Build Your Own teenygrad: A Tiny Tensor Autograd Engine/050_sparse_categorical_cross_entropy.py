def sparse_categorical_cross_entropy(logits, labels):
    if not isinstance(logits, Tensor):
        logits = tensor_from_data(logits)

    log_probs = tensor_log_softmax(logits, axis=-1)
    lp = log_probs.numpy()

    labels_np = np.asarray(labels).astype(int)
    n = lp.shape[0]

    correct = lp[np.arange(n), labels_np]

    loss = -np.mean(correct)
    return tensor_from_data(float(loss))
