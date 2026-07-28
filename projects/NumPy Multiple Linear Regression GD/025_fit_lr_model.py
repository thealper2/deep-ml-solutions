def fit_lr_model(model, X_train, y_train, X_val, y_val):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)
    std[std == 0] = 1.0
    
    X_train_design = prepare_design_matrix(X_train, mean, std)
    X_val_design = prepare_design_matrix(X_val, mean, std)
    
    weights, train_losses, val_losses = train_batch_gd(
        X_train_design, y_train,
        X_val_design, y_val,
        lr=model['learning_rate'],
        epochs=model['epochs'],
        patience=model['patience'],
        seed=model['seed']
    )
    
    XtX = X_train_design.T @ X_train_design
    Xty = X_train_design.T @ y_train
    lambda_reg = 1e-10
    normal_weights = np.linalg.solve(XtX + lambda_reg * np.eye(XtX.shape[0]), Xty)
    
    model['mean'] = mean
    model['std'] = std
    model['weights'] = weights
    model['normal_weights'] = normal_weights
    model['train_losses'] = train_losses
    model['val_losses'] = val_losses
    
    return model
