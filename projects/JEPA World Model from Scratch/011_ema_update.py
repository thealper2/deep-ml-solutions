def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    updated = {}
    for key in target_params:
        updated[key] = tau * target_params[key] + (1 - tau) * encoder_params[key]

    return updated
