def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    n = x_shard.shape[0]
    replace = batch_size > n
    indices = rng.choice(n, size=batch_size, replace=replace)
    return x_shard[indices], y_shard[indices]
