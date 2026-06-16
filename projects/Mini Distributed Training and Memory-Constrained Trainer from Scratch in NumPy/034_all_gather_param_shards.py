def all_gather_param_shards(param_shards_per_worker, shapes, shard_slices_per_worker):
    num_workers = len(param_shards_per_worker)
    if num_workers == 0:
        return {}

    keys = param_shards_per_worker[0].keys()
    result = {}

    for key in keys:
        total_size = np.prod(shapes[key])
        full_flat = np.zeros(total_size, dtype=param_shards_per_worker[0][key].dtype)
        for worker_idx in range(num_workers):
            start, end = shard_slices_per_worker[worker_idx][key]
            full_flat[start:end] = param_shards_per_worker[worker_idx][key]

        result[key] = full_flat.reshape(shapes[key])

    return result
