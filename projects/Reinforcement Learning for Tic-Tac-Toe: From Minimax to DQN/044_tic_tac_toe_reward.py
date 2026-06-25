def tic_tac_toe_reward(game_status, agent_player):
    """Return scalar reward from the agent's perspective.

    game_status: one of 'X_win', 'O_win', 'draw', 'ongoing'.
    agent_player: +1 for X, -1 for O.
    """
    if game_status == 'X_win':
        return 1.0 if agent_player == 1 else -1.0
    elif game_status == 'O_win':
        return 1.0 if agent_player == -1 else -1.0
    else:
        return 0.0
