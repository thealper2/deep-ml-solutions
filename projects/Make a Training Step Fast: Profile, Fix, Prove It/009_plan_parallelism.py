def plan_parallelism(model, cluster):
    """Every tensor-parallel x data-parallel x ZeRO plan that fits, fastest first, and why the others do not."""
    P = model["params_b"] * 1e9
    act_gb = model["act_gb"]
    step_tflop = model["step_tflop"]
    tp_gb = model["tp_gb"]

    gpus = cluster["gpus"]
    gpus_per_node = cluster["gpus_per_node"]
    mem_gb = cluster["mem_gb"]
    tflops = cluster["tflops"]
    intra_gbs = cluster["intra_gbs"]
    inter_gbs = cluster["inter_gbs"]
    tp_penalty = cluster["tp_penalty"]

    fits = []
    rejected = []

    for tp in (1, 2, 4, 8):
        if gpus % tp != 0 or tp > gpus_per_node:
            continue
        dp = gpus // tp
        zero_stages = range(0, 4) if dp > 1 else (0,)
        for zero in zero_stages:
            weights = 2 * P / tp
            gradients = 2 * P / tp
            opt = 12 * P / tp
            if zero >= 3:
                weights /= dp
            if zero >= 2:
                gradients /= dp
            if zero >= 1:
                opt /= dp
            mem = (weights + gradients + opt) / 1e9 + act_gb / tp

            compute = step_tflop / (gpus * tflops) * 1000 * (1 + tp_penalty * (tp - 1))
            tp_time = 0.0 if tp == 1 else tp_gb * 2 * (tp - 1) / tp / intra_gbs * 1000
            if dp == 1:
                dp_time = 0.0
            else:
                B = 2 * P / tp
                dp_time = 2 * (dp - 1) / dp * B / (inter_gbs * 1e9) * 1000
                if zero == 3:
                    dp_time *= 1.5
            step_ms = compute + tp_time + dp_time

            if mem <= mem_gb:
                fits.append({
                    "tp": tp,
                    "dp": dp,
                    "zero": zero,
                    "memory_gb": round(mem, 2),
                    "step_ms": round(step_ms, 2),
                })
            else:
                rejected.append((tp, zero, f"needs {round(mem, 1)} GB"))

    fits.sort(key=lambda p: (p["step_ms"], p["memory_gb"], p["tp"], p["zero"]))
    return {"fits": fits, "rejected": rejected}