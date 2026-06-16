def shard_dataset_across_workers(x, y, num_workers):
    N = x.shape[0]
    shards = []
    base_size = N // num_workers
    remainder = N % num_workers

    start = 0
    for i in range(num_workers):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        shards.append((x[start:end], y[start:end]))
        start = end

    return shards
