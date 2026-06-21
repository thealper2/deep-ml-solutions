def generate_self_play_batch(net, num_games, num_simulations, c_puct, temperature=1.0):
    buffer = []
    for _ in range(num_games):
        history, winner = play_self_play_game(net, num_simulations, c_puct, temperature)
        labelled = assign_value_targets(history, winner)
        buffer.extend(labelled)

    return buffer
