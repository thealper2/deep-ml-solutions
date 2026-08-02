import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_val_bias = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

    n_samples = X_train.shape[0]
    correlations = np.abs(np.corrcoef(X_train.T, y_train)[:-1, -1])
    keep_idx = np.where(correlations > 0.05)[0]

    variances = np.var(X_train, axis=0)
    high_var_idx = np.where(variances > 0.1)[0]
    selected_idx = np.unique(np.concatenate([keep_idx, high_var_idx]))

    if len(selected_idx) < 2:
        selected_idx = np.argsort(correlations)[-10:]

    X_train_sel = X_train[:, selected_idx]
    X_val_sel = X_val[:, selected_idx]
    
    X_train_final = np.hstack([np.ones((X_train_sel.shape[0], 1)), X_train_sel])
    X_val_final = np.hstack([np.ones((X_val.shape[0], 1)), X_val_sel])

    def ridge_fit(X, y, alpha):
        n, d = X.shape
        I = np.eye(d)
        I[0, 0] = 0
        theta = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
        return theta
    
    best_val_rmse = float('inf')
    best_theta = None
    
    for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        theta = ridge_fit(X_train_final, y_train, alpha)
        y_val_pred = X_val_final @ theta
        val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_theta = theta
    
    if best_val_rmse > 0.8:
        for alpha in [1000.0, 2000.0, 5000.0]:
            theta = ridge_fit(X_train_final, y_train, alpha)
            y_val_pred = X_val_final @ theta
            val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_theta = theta
    
    selected_idx_final = selected_idx
    theta_final = best_theta
    
    def predict(X):
        X_sel = X[:, selected_idx_final]
        X_bias = np.hstack([np.ones((X_sel.shape[0], 1)), X_sel])
        return X_bias @ theta_final
    
    return predict
