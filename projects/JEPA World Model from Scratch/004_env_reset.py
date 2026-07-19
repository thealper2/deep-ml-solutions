def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    state = init_env_state(room_size, seed)
    obs = render_observation(state, room_size)
    return state, obs
