def encode_goal(goal_state: torch.Tensor, encoder_params: dict, room_size: int = 8) -> torch.Tensor:
    goal_abs = render_observation(goal_state, room_size)
    goal_emb = encode_batch(goal_abs.unsqueeze(0), encoder_params)
    return goal_emb.squeeze(0)
