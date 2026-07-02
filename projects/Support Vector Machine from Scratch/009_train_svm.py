def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    n_features = x.shape[1]
    params = initialize_parameters(n_features)
    
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    
    return params
