def train(model, loss_fn, optimizer, x, y, epochs, batch_size, seed=0):
    """Run a deterministic minibatch training loop.

    Inputs:
      model: sequential model dict with 'forward', 'backward', 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x: np.ndarray of shape (N, D) training features
      y: np.ndarray of shape (N,) integer class labels
      epochs: int, number of full passes over the data
      batch_size: int, minibatch size
      seed: int, RNG seed for deterministic shuffling / batching

    Returns:
      history: list[float] of length `epochs`; history[t] is the mean
      train_step loss over minibatches in epoch t.
      Model parameters are updated in place; shapes unchanged.
    """
    N = x.shape[0]
    rng = np.random.RandomState(seed)
    history = []

    for epoch in range(epochs):
      indices = np.arange(N)
      rng.shuffle(indices)

      epoch_loss_sum = 0.0
      num_batches = 0

      for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_indices = indices[start:end]

        x_batch = x[batch_indices]
        y_batch = y[batch_indices]

        batch_loss = train_step(model, loss_fn, optimizer, x_batch, y_batch)

        epoch_loss_sum += batch_loss
        num_batches += 1

      avg_epoch_loss = epoch_loss_sum / num_batches
      history.append(avg_epoch_loss)

    return history