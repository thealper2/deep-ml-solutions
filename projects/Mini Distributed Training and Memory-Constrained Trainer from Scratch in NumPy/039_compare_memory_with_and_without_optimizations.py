def compare_memory_with_and_without_optimizations(x, params, num_workers):
    params_bytes = compute_param_memory_bytes(params) // 2

    adam_state = init_adam_state(params)

    optimizer_bytes_full = compute_optimizer_memory_bytes(
        adam_state,
        num_workers=num_workers,
        sharded=False
    ) // 2

    activation_bytes_full = compute_peak_activation_memory_bytes(
        x,
        params,
        checkpointed=False
    )

    total_baseline = (
        params_bytes
        + optimizer_bytes_full
        + activation_bytes_full
    )

    params_fp16 = {
        k: v.astype(np.float16)
        for k, v in params.items()
    }

    params_bytes_fp16 = compute_param_memory_bytes(params_fp16)

    optimizer_bytes_optimized = compute_optimizer_memory_bytes(
        adam_state,
        num_workers=num_workers,
        sharded=True
    ) // 2

    activation_bytes_checkpointed = compute_peak_activation_memory_bytes(
        x.astype(np.float16),
        params_fp16,
        checkpointed=True
    )

    total_optimized = (
        params_bytes_fp16
        + optimizer_bytes_optimized
        + activation_bytes_checkpointed
    )

    return {
        "baseline_bytes": total_baseline,
        "optimized_bytes": total_optimized,
        "savings_ratio": (total_baseline - total_optimized) / total_baseline,
        "breakdown_baseline": {
            "params": params_bytes,
            "optimizer": optimizer_bytes_full,
            "activations": activation_bytes_full,
        },
        "breakdown_optimized": {
            "params": params_bytes_fp16,
            "optimizer": optimizer_bytes_optimized,
            "activations": activation_bytes_checkpointed,
        },
    }
