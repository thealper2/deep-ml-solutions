def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    rng = np.random.default_rng(0)
    loss_history = []
    current_w = w.copy()

    for step in range(num_steps):
        x_batch, y_batch = get_batch(data, block_size, batch_size, rng)
        ids = x_batch.flatten()
        targets = y_batch.flatten()
        result = run_one_training_step(current_w, ids, targets, learning_rate)
        current_w = result['w']
        if step % log_every == 0:
            loss_history.append(result['loss'])

    return {'w': current_w, 'loss_history': loss_history}
