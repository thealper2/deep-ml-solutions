def logging_and_validation_loss(params, val_ids, block_size, batch_size, n_eval_batches):
    """Estimate validation cross-entropy loss by averaging over several batches."""
    rng = np.random.default_rng(42)
    total_loss = 0.0

    for _ in range(n_eval_batches):
        X_batch, Y_batch = get_batch(val_ids, block_size, batch_size, rng)

        logits, _ = full_model_forward(X_batch, params)

        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = Y_batch.reshape(-1)

        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(targets_flat)), targets_flat] = 1.0
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-15), axis=1))

        total_loss += loss

    return total_loss / n_eval_batches
