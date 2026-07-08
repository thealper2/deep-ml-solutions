def classification_accuracy(params, x, y):
    logits, _ = model_forward(params, x)
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == y))
