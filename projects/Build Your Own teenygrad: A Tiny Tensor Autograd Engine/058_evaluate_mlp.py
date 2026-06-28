def evaluate_mlp(model, X_test, y_test):
    X_tensor = tensor_from_data(np.asarray(X_test, dtype=np.float32))
    logits = model(X_tensor)
    return accuracy(logits, y_test)
