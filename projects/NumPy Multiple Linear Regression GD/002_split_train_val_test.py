def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
