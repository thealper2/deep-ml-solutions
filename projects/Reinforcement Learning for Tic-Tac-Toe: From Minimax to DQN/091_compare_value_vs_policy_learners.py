def compare_value_vs_policy_learners(num_episodes=10, eval_games=2, seed=0):
    """Train Q-learning, SARSA, REINFORCE under matched settings; return per-agent dicts."""
    num_episodes = int(num_episodes)
    eval_games = int(eval_games)

    rng = np.random.default_rng(seed)
    result = {}

    alpha = 0.1
    gamma = 0.9
    epsilon_start = 1.0
    epsilon_min = 0.05
    decay_rate = 0.001
    hidden_dim = 32

    def random_opponent_flat(board, player, rng):
        row, col = random_move_agent(board, player, rng)
        return row * 3 + col

    rng_q = np.random.default_rng(seed + 1)
    q_res = train_q_learning_agent(
        num_episodes, alpha, gamma, epsilon_start, epsilon_min, decay_rate,
        random_opponent_flat, rng_q
    )
    q_table = q_res['q_table']

    rng_eval = np.random.default_rng(seed + 100)
    q_vs_random = evaluate_q_agent_vs_random(q_table, eval_games, rng_eval)
    q_vs_minimax = evaluate_q_agent_vs_minimax(q_table, eval_games, rng_eval)

    rng_s = np.random.default_rng(seed + 2)
    sarsa_res = train_sarsa_agent(
        num_episodes, alpha, gamma, epsilon_start, epsilon_min, decay_rate,
        random_opponent_flat, rng_s
    )
    sarsa_table = sarsa_res['q_table']

    sarsa_vs_random = evaluate_q_agent_vs_random(sarsa_table, eval_games, rng_eval)
    sarsa_vs_minimax = evaluate_q_agent_vs_minimax(sarsa_table, eval_games, rng_eval)

    rng_r = np.random.default_rng(seed + 3)
    reinforce_res = train_reinforce_agent(
        num_episodes, gamma, 1e-2, hidden_dim,
        random_opponent_flat, rng_r, init_seed=seed + 4
    )
    reinforce_params = reinforce_res['params']
    reinforce_outcomes = reinforce_res['episode_outcomes']

    reinforce_scores = []
    for outcome in reinforce_outcomes:
        if outcome == 'X_win':
            reinforce_scores.append(1.0)
        elif outcome == 'O_win':
            reinforce_scores.append(-1.0)
        else:
            reinforce_scores.append(0.0)

    def reinforce_greedy_agent(board, player, rng):
        state = encode_board_flat_length_nine(board, player)
        legal_mask = np.zeros(9, dtype=bool)
        for row, col in get_legal_moves(board):
            legal_mask[row * 3 + col] = True
        q_values, _ = mlp_forward_pass(reinforce_params, state.reshape(1, -1))
        masked_q = mask_illegal_actions_neg_inf(q_values.flatten(), legal_mask)
        return int(np.argmax(masked_q))

    reinforce_wins = 0
    reinforce_losses = 0
    reinforce_draws = 0

    for game_idx in range(eval_games):
        game = TicTacToeGame()
        agent_player = 1 if game_idx % 2 == 0 else -1

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                action = reinforce_greedy_agent(board, current_player, rng_eval)
                out = episode_apply_action(board, action, current_player, agent_player)
            else:
                move = random_move_agent(board, current_player, rng_eval)
                if isinstance(move, tuple):
                    row, col = move
                else:
                    row = move // 3
                    col = move % 3
                out = episode_apply_action(board, row * 3 + col, current_player, agent_player)

            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']

        status = game.status
        if status == 'X_win':
            if agent_player == 1:
                reinforce_wins += 1
            else:
                reinforce_losses += 1
        elif status == 'O_win':
            if agent_player == -1:
                reinforce_wins += 1
            else:
                reinforce_losses += 1
        else:
            reinforce_draws += 1

    reinforce_vs_random = {
        'win_rate': reinforce_wins / eval_games if eval_games > 0 else 0.0,
        'loss_rate': reinforce_losses / eval_games if eval_games > 0 else 0.0,
        'draw_rate': reinforce_draws / eval_games if eval_games > 0 else 0.0
    }

    reinforce_wins_mm = 0
    reinforce_losses_mm = 0
    reinforce_draws_mm = 0

    for game_idx in range(eval_games):
        game = TicTacToeGame()
        agent_player = 1 if game_idx % 2 == 0 else -1

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                action = reinforce_greedy_agent(board, current_player, rng_eval)
                out = episode_apply_action(board, action, current_player, agent_player)
            else:
                _, move = minimax_alpha_beta(board, current_player, -float('inf'), float('inf'))
                row, col = move
                out = episode_apply_action(board, row * 3 + col, current_player, agent_player)

            game.board = out['next_board']
            game.current_player = out['next_player']
            game.status = out['status']

        status = game.status
        if status == 'X_win':
            if agent_player == 1:
                reinforce_wins_mm += 1
            else:
                reinforce_losses_mm += 1
        elif status == 'O_win':
            if agent_player == -1:
                reinforce_wins_mm += 1
            else:
                reinforce_losses_mm += 1
        else:
            reinforce_draws_mm += 1

    reinforce_vs_minimax = {
        'win_rate': reinforce_wins_mm / eval_games if eval_games > 0 else 0.0,
        'loss_rate': reinforce_losses_mm / eval_games if eval_games > 0 else 0.0,
        'draw_rate': reinforce_draws_mm / eval_games if eval_games > 0 else 0.0
    }

    q_outcomes = q_res['episode_outcomes']
    q_scores = []
    for outcome in q_outcomes:
        if outcome == 'X_win':
            q_scores.append(1.0)
        elif outcome == 'O_win':
            q_scores.append(-1.0)
        else:
            q_scores.append(0.0)

    sarsa_outcomes = sarsa_res['episode_outcomes']
    sarsa_scores = []
    for outcome in sarsa_outcomes:
        if outcome == 'X_win':
            sarsa_scores.append(1.0)
        elif outcome == 'O_win':
            sarsa_scores.append(-1.0)
        else:
            sarsa_scores.append(0.0)

    result['q_learning'] = {
        'win_rate_vs_random': q_vs_random['win_rate'],
        'draw_rate_vs_minimax': q_vs_minimax['draw_rate'],
        'learning_curve': q_scores
    }

    result['sarsa'] = {
        'win_rate_vs_random': sarsa_vs_random['win_rate'],
        'draw_rate_vs_minimax': sarsa_vs_minimax['draw_rate'],
        'learning_curve': sarsa_scores
    }

    result['reinforce'] = {
        'win_rate_vs_random': reinforce_vs_random['win_rate'],
        'draw_rate_vs_minimax': reinforce_vs_minimax['draw_rate'],
        'learning_curve': reinforce_scores
    }

    return result
