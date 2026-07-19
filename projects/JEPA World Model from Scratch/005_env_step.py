def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    next_state = apply_action(state, action, room_size)
    next_obs = render_observation(next_state, room_size)
    return next_state, next_obs
