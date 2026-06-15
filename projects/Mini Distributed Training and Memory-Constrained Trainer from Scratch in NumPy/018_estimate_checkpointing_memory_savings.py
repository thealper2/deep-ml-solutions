def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    full_bytes = (batch_size * in_dim + 2 * batch_size * hidden_dim) * dtype_bytes
    checkpoint_bytes = (batch_size * in_dim) * dtype_bytes
    saved_bytes = full_bytes - checkpoint_bytes
    return {
        'full_bytes': full_bytes,
        'checkpoint_bytes': checkpoint_bytes,
        'saved_bytes': saved_bytes,
    }
