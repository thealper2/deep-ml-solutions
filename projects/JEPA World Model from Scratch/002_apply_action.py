def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    x, y = state[0].item(), state[1].item()

    if action == 0:
        y -= 1
    elif action == 1:
        y += 1
    elif action == 2:
        x -= 1
    elif action == 3:
        x += 1

    x = max(0, min(room_size - 1, x))
    y = max(0, min(room_size - 1, y))
    return torch.tensor([float(x), float(y)])
