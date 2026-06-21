def run_mcts(state, to_play, net, num_simulations, c_puct):
    root = make_mcts_node()
    root['board'] = state
    root['to_play'] = to_play

    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)

    return root
