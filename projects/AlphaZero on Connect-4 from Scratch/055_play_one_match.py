def play_one_match(agent_one, agent_two, starting_player=1):
    board = make_empty_board()
    to_play = starting_player
    done = False
    winner = 0

    while not done:
        if to_play == 1:
            action = agent_one(board, to_play)
        else:
            action = agent_two(board, to_play)

        board, done, winner, to_play = step_env(board, action, to_play)

    return winner
