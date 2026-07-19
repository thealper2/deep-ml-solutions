def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    x = int(state[0].item())
    y = int(state[1].item())
    obs = torch.zeros((1, room_size, room_size), dtype=torch.float32)
    obs[0, y, x] = 1.0
    return obs
