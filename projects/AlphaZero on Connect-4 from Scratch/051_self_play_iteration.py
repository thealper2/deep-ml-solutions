def self_play_iteration(net, optimizer, num_games, num_simulations, c_puct, batch_size, num_epochs=1, temperature=1.0):
    buffer = generate_self_play_batch(net, num_games, num_simulations, c_puct, temperature)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = training_epoch(net, optimizer, buffer, batch_size, seed=epoch)
        losses.append(epoch_loss)

    return {
        'buffer_size': len(buffer),
        'losses': losses,
    }
