def full_distributed_training_loop(x, y, num_workers=2, num_steps=10, micro_batch_size=8, lr=1e-3, hidden_dim=16, use_checkpointing=True, use_mixed_precision=True, use_zero=True, seed=0):
    np.random.seed(seed)
    in_dim = x.shape[1]
    out_dim = y.shape[1]
    params = init_mlp_params(in_dim, hidden_dim, out_dim, seed)

    adam_state = init_adam_state(params)
    worker_states = partition_optimizer_state(adam_state, num_workers)

    loss_history = []

    for step in range(num_steps):
        shards = shard_dataset_across_workers(x, y, num_workers)
        per_worker_grads = []

        for worker_idx in range(num_workers):
            x_shard, y_shard = shards[worker_idx]
            micro_batches = split_into_micro_batches(x_shard, y_shard, micro_batch_size)
            accum_grads = None

            for x_mb, y_mb in micro_batches:
                if use_mixed_precision:
                    x_mb_fp16 = x_mb.astype(np.float16)
                    y_mb_fp16 = y_mb.astype(np.float16)

                    fp16_params = {}
                    for key in params:
                        fp16_params[key] = params[key].astype(np.float16)

                    y_pred, cache = mlp_forward(x_mb_fp16, fp16_params)
                    _, dy_pred = mse_loss_and_grad(y_pred, y_mb_fp16)

                    scale = 128.0
                    dy_pred_scaled = dy_pred * scale

                    if use_checkpointing:
                        light_cache = {'x': x_mb_fp16}
                        grads_mb = mlp_backward_checkpointed(dy_pred_scaled, light_cache, fp16_params)
                    else:
                        grads_mb = mlp_backward(dy_pred_scaled, cache, fp16_params)

                    grads_mb_fp32 = {}
                    for key in grads_mb:
                        grads_mb_fp32[key] = grads_mb[key].astype(np.float32) / scale

                else:
                    y_pred, cache = mlp_forward(x_mb, params)
                    _, dy_pred = mse_loss_and_grad(y_pred, y_mb)

                    if use_checkpointing:
                        light_cache = {'x': x_mb}
                        grads_mb = mlp_backward_checkpointed(dy_pred, light_cache, params)
                    else:
                        grads_mb = mlp_backward(dy_pred, cache, params)

                    grads_mb_fp32 = grads_mb
                
                accum_grads = accumulate_gradients(accum_grads, grads_mb_fp32)

            num_micro = len(micro_batches)
            if num_micro > 0:
                scaled_grads = scale_accumulated_gradients(accum_grads, num_micro)
            else:
                scaled_grads = accum_grads
            
            per_worker_grads.append(scaled_grads)

        avg_grads = all_reduce_mean(per_worker_grads)

        if use_zero:
            params, worker_states = zero_optimizer_step(params, avg_grads, worker_states, lr=lr)
        else:
            for key in params:
                params[key] -= lr * avg_grads[key]

        y_pred, _ = mlp_forward(x, params)
        loss, _ = mse_loss_and_grad(y_pred, y)
        loss_history.append(float(loss))

    return {'loss_history': loss_history, 'final_params': params}
