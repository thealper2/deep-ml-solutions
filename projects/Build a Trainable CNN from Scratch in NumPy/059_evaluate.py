def evaluate(params, x, y):
    preds = lenet_predict(x, params)
    return np.mean(preds == y)
