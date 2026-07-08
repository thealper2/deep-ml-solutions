def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    class_to_worker = {}
    for c in range(num_classes):
        class_to_worker[c] = c % num_workers

    worker_indices = [[] for _ in range(num_workers)]
    for idx, label in enumerate(y):
        worker = class_to_worker[label]
        worker_indices[worker].append(idx)

    rng = np.random.default_rng(seed)
    shards = []
    for worker in range(num_workers):
        indices = np.array(worker_indices[worker])
        rng.shuffle(indices)
        shards.append((x[indices], y[indices]))

    return shards
