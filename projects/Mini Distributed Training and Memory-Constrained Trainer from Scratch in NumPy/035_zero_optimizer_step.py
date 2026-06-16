def zero_optimizer_step(params, grads, worker_states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    num_workers = len(worker_states)
    updated_shards_per_worker = []
    for worker_idx in range(num_workers):
        updated_shards, updated_state = local_shard_adam_update(
            params, grads, worker_states[worker_idx], lr, beta1, beta2, eps
        )
        updated_shards_per_worker.append(updated_shards)
        worker_states[worker_idx] = updated_state

    shapes = worker_states[0]['shapes']
    shard_slices_per_worker = [ws['shard_slices'] for ws in worker_states]

    new_params = all_gather_param_shards(
        updated_shards_per_worker, shapes, shard_slices_per_worker
    )
    return new_params, worker_states
