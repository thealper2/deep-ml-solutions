def prepare_design_matrix(X, mean, std):
    X_std = (X - mean) / std
    return add_bias_column(X_std)
