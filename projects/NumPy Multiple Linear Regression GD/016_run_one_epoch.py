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
