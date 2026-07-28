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
