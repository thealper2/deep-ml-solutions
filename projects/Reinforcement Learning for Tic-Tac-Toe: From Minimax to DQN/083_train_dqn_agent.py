def train_dqn_agent(num_episodes, hidden_dim=64, gamma=0.99, lr=1e-3, batch_size=64, buffer_capacity=10000, sync_every_k=200, epsilon_start=1.0, epsilon_end=0.05, seed=0):
    """Full DQN self-play training loop. Returns dict with online_params,
    target_params, loss_history, reward_history, architecture."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    arch = build_mlp_architecture(9, hidden_dim, 9)
    online_params = initialize_mlp_parameters(arch, seed=seed)
    target_params = build_target_network_copy(online_params)
    adam_state = {}
    buffer = create_replay_buffer(buffer_capacity)

    loss_history = []
    reward_history = []

    eps = epsilon_start

    for episode in range(num_episodes):
        eps = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / max(1, num_episodes)))

        game = TicTacToeGame()
        agent_player = 1

        episode_reward = 0.0
        step_count = 0
        episode_loss_sum = 0.0
        episode_loss_count = 0

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            state = encode_board_flat_length_nine(board, current_player)
            legal_mask = np.zeros(9, dtype=bool)
            for row, col in get_legal_moves(board):
                legal_mask[row * 3 + col] = True

            action = dqn_select_action(online_params, state, legal_mask, eps, rng)

            out = episode_apply_action(board, action, current_player, current_player)
            next_board = out['next_board']
            status = out['status']
            reward = float(out['reward'])
            done = out['done']
            next_player = out['next_player'] if not done else current_player

            next_state = encode_board_flat_length_nine(next_board, next_player)
            next_legal_mask = np.zeros(9, dtype=bool)
            for row, col in get_legal_moves(next_board):
                next_legal_mask[row * 3 + col] = True

            append_transition_to_buffer(buffer, state, action, reward, next_state, done, next_legal_mask)

            episode_reward += reward

            game.board = next_board
            game.current_player = next_player if not done else current_player
            game.status = status

            step_count += 1

            if len(buffer['data']) >= batch_size:
                online_params, adam_state, loss = dqn_train_step(
                    online_params, target_params, adam_state, buffer, batch_size, gamma, lr, rng
                )
                episode_loss_sum += loss
                episode_loss_count += 1

            target_params = sync_target_network_periodically(
                online_params, target_params, step_count, sync_every_k
            )

        reward_history.append(episode_reward)
        episode_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0.0
        loss_history.append(episode_loss)

    return {
        'online_params': online_params,
        'target_params': target_params,
        'loss_history': loss_history,
        'reward_history': reward_history,
        'architecture': arch
    }
