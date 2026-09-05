def legal_actions(pos: tuple, G: int) -> list:
    actions = []
    for action in range(4):
        _, legal = grid_step(pos, action, G)
        if legal:
            actions.append(action)

    return actions