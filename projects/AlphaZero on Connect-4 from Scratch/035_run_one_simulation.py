def run_one_simulation(root, net, c_puct):
    leaf = select_leaf(root, c_puct)

    board = leaf['board']
    to_play = leaf['to_play']
    done, winner = is_terminal(board)

    if done:
        if winner == 0:
            value = 0.0
        elif winner == to_play:
            value = 1.0
        else:
            value = -1.0

        backup_value(leaf, value)

    else:
        priors, value = evaluate_with_network(net, board, to_play)
        expand_node(leaf, priors)
        backup_value(leaf, value)
