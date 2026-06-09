def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Run num_epochs full-batch SGD updates and return the final params."""
    current_params = params
    for _ in range(num_epochs):
        current_params, _ = training_step(current_params, x, one_hot_targets, learning_rate)

    return current_params
