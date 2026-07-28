def learning_curve_data(train_losses, val_losses):
    epochs = list(range(1, len(train_losses) + 1))
    train = train_losses.tolist() if isinstance(train_losses, np.ndarray) else train_losses
    val = val_losses.tolist() if isinstance(val_losses, np.ndarray) else val_losses
    return epochs, train, val
