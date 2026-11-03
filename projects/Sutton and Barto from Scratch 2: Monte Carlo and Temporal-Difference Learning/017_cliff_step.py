def cliff_step(state, action):
    state = int(state)
    action = int(action)

    row, col = divmod(state, 12)

    if action == 0:
        row = max(0, row - 1)
    elif action == 1:
        col = min(11, col + 1)
    elif action == 2:
        row = min(3, row + 1)
    elif action == 3:
        col = max(0, col - 1)

    next_state = row * 12 + col

    if 37 <= next_state <= 46:
        return 36, -100.0, False

    if next_state == 47:
        return 47, -1.0, True

    return int(next_state), -1.0, False