def play_self_play_game(net, num_simulations, c_puct, temperature=1.0):
    board = make_empty_board()
    to_play = 1
    history = []
    done = False
    winner = 0

    while not done:
        action, policy = mcts_choose_action(board, to_play, net, num_simulations, c_puct, temperature)
        record_self_play_step(history, board, policy, to_play)
        board, done, winner, to_play = step_env(board, action, to_play)

    return history, winner
