def init_target_encoder(encoder_params: dict) -> dict:
    target_params = {}
    for key, value in encoder_params.items():
        target_params[key] = value.detach().clone()

    return target_params
