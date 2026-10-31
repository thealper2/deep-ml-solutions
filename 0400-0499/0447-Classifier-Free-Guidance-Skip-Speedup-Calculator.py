import numpy as np

def cfg_skip_speedup(
    T: int,
    skip_mask: list[bool],
    time_per_pass: float,
    cond_preds: list[float],
    uncond_preds: list[float],
    guidance_scale: float
) -> dict:
    """
    Analyze the speedup and quality impact of skipping unconditional
    passes in Classifier-Free Guidance diffusion inference.
    
    Args:
        T: Total number of denoising timesteps
        skip_mask: Boolean list; True means skip the unconditional pass at that step
        time_per_pass: Time in milliseconds for a single forward pass
        cond_preds: Conditional model predictions at each timestep
        uncond_preds: Unconditional model predictions at each timestep
        guidance_scale: CFG guidance scale (w)
    
    Returns:
        Dictionary with speedup metrics and guided outputs
    """
    standard = []
    skipped = []
    cache = 0.0
    passes_standard = 0
    passes_skipped = 0

    for t in range(T):
        c = cond_preds[t]
        u = uncond_preds[t]

        passes_standard += 2
        standard.append(u + guidance_scale * (c - u))

        if skip_mask[t]:
            passes_skipped += 1
            u_used = cache
        else:
            passes_skipped += 2
            cache = u
            u_used = u
            
        skipped.append(u_used + guidance_scale * (c - u_used))

    total_standard = passes_standard
    total_skipped = passes_skipped
    speedup = round(total_standard / total_skipped, 4) if total_skipped else 0.0
    time_saved = round((total_standard - total_skipped) * time_per_pass, 4)
    max_dev = round(max(abs(a - b) for a, b in zip(standard, skipped)), 4)

    return {
        'total_passes_standard': total_standard,
        'total_passes_skipped': total_skipped,
        'speedup_ratio': speedup,
        'time_saved_ms': time_saved,
        'guided_outputs_standard': [round(v, 4) for v in standard],
        'guided_outputs_skipped': [round(v, 4) for v in skipped],
        'max_output_deviation': max_dev,
    }