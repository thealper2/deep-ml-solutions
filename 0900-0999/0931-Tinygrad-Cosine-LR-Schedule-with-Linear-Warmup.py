import math

def cosine_with_warmup(step, warmup_steps, total_steps, base_lr):
    if step < 0 or step > total_steps:
        return 0.0

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step / warmup_steps)

    if warmup_steps <= 0:
        warmup_steps = 0

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * cosine_factor