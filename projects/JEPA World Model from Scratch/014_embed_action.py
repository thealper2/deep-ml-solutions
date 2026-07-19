def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    action_embed_w = predictor_params['action_embed_w']
    return action_embed_w[actions]
