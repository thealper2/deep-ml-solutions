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
