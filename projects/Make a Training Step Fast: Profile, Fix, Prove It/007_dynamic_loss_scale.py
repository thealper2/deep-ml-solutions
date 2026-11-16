def dynamic_loss_scale(grad_stats, init_scale=65536.0, growth_interval=4, growth=2.0, backoff=0.5,
                       fp16_max=65504.0, fp16_min=2.0 ** -24):
    """Simulate a dynamic fp16 loss scaler: skip and back off on overflow, grow after clean steps."""
    fp16_max = 65504.0
    fp16_min = 2.0 ** -24

    scale = float(init_scale)
    scales = []
    skipped = []
    underflow_count = 0
    clean = 0

    for step, (largest, smallest) in enumerate(grad_stats):
        scales.append(scale)
        if largest * scale > fp16_max:
            skipped.append(step)
            scale *= backoff
            clean = 0
            continue
        if smallest * scale < fp16_min:
            underflow_count += 1
        clean += 1
        if clean >= growth_interval:
            scale *= growth
            clean = 0

    return {
        "scales": scales,
        "skipped": skipped,
        "underflow_steps": underflow_count,
        "final_scale": scale,
    }