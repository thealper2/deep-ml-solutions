def random_walk_to_goal(start: tuple, goal: tuple, G: int, max_len: int, rng) -> list:
    pos = start
    moves = []

    for _ in range(max_len):
        if pos == goal:
            break

        legal = legal_actions(pos, G)
        action = rng.choice(legal)
        pos, _ = grid_step(pos, action, G)
        moves.append(int(action))

    return moves