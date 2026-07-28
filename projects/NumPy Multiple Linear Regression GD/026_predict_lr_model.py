def predict_lr_model(model, X):
    X_std = (X - model['mean']) / model['std']
    X_design = add_bias_column(X_std)
    return X_design @ model['weights']
