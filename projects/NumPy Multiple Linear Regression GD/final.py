"""
NumPy Multiple Linear Regression GD — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  shuffle_xy ──
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    X_ = X[indices, :]
    y_ = y[indices]
    return X_, y_

# ── Step 002  split_train_val_test ──
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

# ── Step 003  compute_feature_stats ──
def compute_feature_stats(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std

# ── Step 004  standardize_features ──
def standardize_features(X, mean, std):
    return (X - mean) / std

# ── Step 005  add_bias_column ──
def add_bias_column(X):
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=X.dtype)
    return np.hstack([ones, X])

# ── Step 006  prepare_design_matrix ──
def prepare_design_matrix(X, mean, std):
    X_std = (X - mean) / std
    return add_bias_column(X_std)

# ── Step 007  predict_linear ──
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    return X @ weights

# ── Step 008  mse_loss ──
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ── Step 009  mse_gradient ──
def mse_gradient(X, y_true, y_pred):
    N = X.shape[0]
    return (2 / N) * (X.T @ (y_pred - y_true))

# ── Step 010  normal_equation ──
def normal_equation(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

# ── Step 011  initialize_weights ──
def initialize_weights(n_features, seed=None):
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(0, 0.01, n_features)

# ── Step 012  gd_step ──
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    N = X.shape[0]
    grad = (2 / N) * X.T @ (X @ weights - y)
    return weights - lr * grad

# ── Step 013  epoch_train_val_losses ──
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    train_pred = X_train @ weights
    val_pred = X_val @ weights
    train_loss = float(np.mean((train_pred - y_train) ** 2))
    val_loss = float(np.mean((val_pred - y_val) ** 2))
    return train_loss, val_loss

# ── Step 014  update_early_stop_state ──
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    if val_loss < best_val_loss:
        return val_loss, 0, weights.copy(), False
    else:
        wait += 1
        if wait >= patience:
            return best_val_loss, wait, best_weights, True

        return best_val_loss, wait, best_weights, False

# ── Step 015  init_training_state ──
def init_training_state(n_features, seed=None):
    weights = initialize_weights(n_features, seed)
    return {
        'weights': weights,
        'best_weights': weights.copy(),
        'best_val_loss': np.inf,
        'wait': 0,
        'train_losses': [],
        'val_losses': [],
        'stopped': False,
    }

# ── Step 016  run_one_epoch ──
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    state['weights'] = gd_step(X_train, y_train, state['weights'], lr)
    
    train_loss, val_loss = epoch_train_val_losses(
        X_train, y_train, X_val, y_val, state['weights']
    )
    
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)
    
    best_val_loss, wait, best_weights, stopped = update_early_stop_state(
        val_loss, state['best_val_loss'], state['wait'], 
        state['weights'], state['best_weights'], patience
    )
    
    state['best_val_loss'] = best_val_loss
    state['wait'] = wait
    state['best_weights'] = best_weights
    state['stopped'] = stopped
    
    return state

# ── Step 017  train_batch_gd ──
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    n_features = X_train.shape[1]
    state = init_training_state(n_features, seed)

    for _ in range(epochs):
        if state['stopped']:
            break

        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)

    return state['best_weights'], state['train_losses'], state['val_losses']

# ── Step 018  mean_absolute_error ──
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ── Step 019  root_mean_squared_error ──
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ── Step 020  r_squared ──
def r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
        
    r_s = 1 - (ss_res / ss_tot)
    return r_s

# ── Step 021  evaluate_regression ──
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

# ── Step 022  learning_curve_data ──
def learning_curve_data(train_losses, val_losses):
    epochs = list(range(1, len(train_losses) + 1))
    train = train_losses.tolist() if isinstance(train_losses, np.ndarray) else train_losses
    val = val_losses.tolist() if isinstance(val_losses, np.ndarray) else val_losses
    return epochs, train, val

# ── Step 023  weights_l2_distance ──
def weights_l2_distance(w_gd, w_closed):
    return float(np.linalg.norm(w_gd - w_closed))

# ── Step 024  create_lr_model ──
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    return {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience,
        'seed': seed,
        'weights': None,
        'normal_weights': None,
        'mean': None,
        'std': None,
        'train_losses': [],
        'val_losses': []
    }

# ── Step 025  fit_lr_model ──
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

# ── Step 026  predict_lr_model ──
def predict_lr_model(model, X):
    X_std = (X - model['mean']) / model['std']
    X_design = add_bias_column(X_std)
    return X_design @ model['weights']

# ── Step 027  score_lr_model ──
import numpy as np

def score_lr_model(model, X, y):
    y_pred = predict_lr_model(model, X)
    return evaluate_regression(y, y_pred)

# ── Step 028  compare_with_normal_equation ──
def compare_with_normal_equation(model):
    return weights_l2_distance(model['weights'], model['normal_weights'])

# ── Scaffold (runner) ──
"""Demo: from-scratch multiple linear regression with batch GD in NumPy."""
import numpy as np


def main():
    np.random.seed(0)
    n_samples, n_features = 150, 3
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -2.0, 0.5])
    y = X @ true_weights + 0.3 + 0.1 * np.random.randn(n_samples)

    X, y = shuffle_xy(X, y, seed=42)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
        X, y, train_frac=0.6, val_frac=0.2
    )
    print("Splits:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    model = create_lr_model(learning_rate=0.05, epochs=400, patience=25, seed=0)
    model = fit_lr_model(model, X_train, y_train, X_val, y_val)

    y_hat = predict_lr_model(model, X_test[:5])
    print("Sample preds:", np.round(y_hat, 4))
    print("Sample trues:", np.round(y_test[:5], 4))

    metrics = score_lr_model(model, X_test, y_test)
    print("Test MAE/RMSE/R2:", metrics)

    gap = compare_with_normal_equation(model)
    print("GD vs normal-eq L2 gap:", float(gap))

    train_losses = model.get("train_losses", [])
    val_losses = model.get("val_losses", [])
    if len(train_losses) > 0:
        epochs, tr, va = learning_curve_data(train_losses, val_losses)
        print("Final train/val MSE:", float(tr[-1]), float(va[-1]))
        print("Epochs run:", int(epochs[-1]) + 1 if len(epochs) else 0)


if __name__ == "__main__":
    main()
