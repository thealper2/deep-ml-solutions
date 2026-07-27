def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    N, D = X.shape
    w, b = initialize_logistic_params(D)
    losses = []
    
    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(X, y, w, b, lr, l2_lambda)
        losses.append(loss)
    
    return w, b, losses
