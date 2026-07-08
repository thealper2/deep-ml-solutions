def iid_shard_dataset(x, y, num_workers, seed=0):
    np.random.seed(seed)
    N = x.shape[0]
    indices = np.random.permutation(N)

    shards = []
    base_size = N // num_workers
    remainder = N % num_workers

    start = 0
    for i in range(num_workers):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        shard_indices = indices[start:end]
        shards.append((x[shard_indices], y[shard_indices]))
        start = end

    return shards
