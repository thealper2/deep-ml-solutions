def train_loop(net, optimizer, num_iterations, num_games, num_simulations, c_puct, batch_size, num_epochs=1, temperature=1.0):
    history = []
    for _ in range(num_iterations):
        result = self_play_iteration(
            net, optimizer, num_games, num_simulations, c_puct,
            batch_size, num_epochs, temperature
        )
        history.append(result)

    return history
