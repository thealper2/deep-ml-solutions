def train_mlp(X, y, epochs=50, learning_rate=0.1, hidden=16, seed=0):
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y).astype(np.int64)

    in_features = X_np.shape[1]
    out_features = int(y_np.max()) + 1

    model = MLP(in_features, hidden, out_features, seed=seed)
    X_tensor = tensor_from_data(X_np)

    loss_history = []

    for _ in range(epochs):
        logits = model(X_tensor)
        loss = sparse_categorical_cross_entropy(logits, y_np)

        zero_grad(model.parameters())
        tensor_backward(loss)
        sgd_step(model.parameters(), learning_rate)

        loss_history.append(float(loss.numpy()))

    return model, loss_history
