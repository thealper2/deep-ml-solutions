def data_parallel_train_step(x, y, params, num_workers, lr):
    shards = shard_dataset_across_workers(x, y, num_workers)
    per_worker_grads = []
    for x_shard, y_shard in shards:
        grads = compute_local_gradients(x_shard, y_shard, params)
        per_worker_grads.append(grads)

    avg_grads = all_reduce_mean(per_worker_grads)
    new_params = {}
    for key in params:
        new_params[key] = params[key] - lr * avg_grads[key]

    return new_params
