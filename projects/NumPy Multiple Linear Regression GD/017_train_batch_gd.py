def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    n_features = X_train.shape[1]
    state = init_training_state(n_features, seed)

    for _ in range(epochs):
        if state['stopped']:
            break

        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)

    return state['best_weights'], state['train_losses'], state['val_losses']
