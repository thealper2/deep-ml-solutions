def accuracy(logits, labels):
    if isinstance(logits, Tensor):
        logits_np = logits.numpy()
    else:
        logits_np = np.asarray(logits)

    preds = np.argmax(logits_np, axis=1)
    labels_np = np.asarray(labels)

    return float(np.mean(preds == labels_np))
