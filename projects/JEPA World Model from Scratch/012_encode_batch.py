def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    return encoder_forward(obs, encoder_params)
