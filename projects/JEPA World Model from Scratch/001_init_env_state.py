def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed),

    x = torch.randint(0, room_size, (1,)).float()
    y = torch.randint(0, room_size, (1,)).float()
    return torch.cat([x, y])
