def all_reduce_mean(per_worker_grads):
    num_workers = len(per_worker_grads)
    if num_workers == 0:
        return {}

    result = {}
    keys = per_worker_grads[0].keys()

    for key in keys:
        stacked = np.stack([g[key] for g in per_worker_grads])
        result[key] = np.mean(stacked, axis=0)

    return result
