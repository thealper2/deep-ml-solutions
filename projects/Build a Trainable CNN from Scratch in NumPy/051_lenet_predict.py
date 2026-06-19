def lenet_predict(x, params):
    logits, _ = lenet_forward(x, params)
    return np.argmax(logits, axis=1)
