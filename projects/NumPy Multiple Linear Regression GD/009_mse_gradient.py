def mse_gradient(X, y_true, y_pred):
    N = X.shape[0]
    return (2 / N) * (X.T @ (y_pred - y_true))
