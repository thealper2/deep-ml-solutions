def compare_dqn_tabular_random_minimax(dqn_artifacts, q_table, num_games=1, seed=42):
    """Round-robin evaluation among DQN, tabular Q, random, and minimax agents."""
    rng = np.random.default_rng(seed)

    online_params = dqn_artifacts['online_params']

    def dqn_move(board, player):
        state = encode_board_flat_length_nine(board, player)
        legal_mask = np.zeros(9, dtype=bool)
        for row, col in get_legal_moves(board):
            legal_mask[row * 3 + col] = True
        return dqn_select_action(online_params, state, legal_mask, 0.0, rng)

    def tabular_move(board, player):
        state_key = canonical_board_key(board)
        legal_actions = [row * 3 + col for row, col in get_legal_moves(board)]
        return greedy_argmax_over_legal_actions(q_table, state_key, legal_actions, rng)

    def random_move(board, player):
        row, col = random_move_agent(board, player, rng)
        return row * 3 + col

    def minimax_move(board, player):
        _, move = minimax_max_min_step(board, player)
        row, col = move
        return row * 3 + col

    agents = {
        'dqn': dqn_move,
        'tabular': tabular_move,
        'random': random_move,
        'minimax': minimax_move,
    }

    matchups = [
        ('dqn', 'random'),
        ('dqn', 'minimax'),
        ('dqn', 'tabular'),
        ('tabular', 'random'),
        ('tabular', 'minimax'),
        ('random', 'minimax'),
    ]

    def play_matchup(agent_a, agent_b):
        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(num_games):
            game = TicTacToeGame()
            # Alternate which agent plays first as X.
            if game_idx % 2 == 0:
                x_agent, o_agent = agent_a, agent_b
                a_is_x = True
            else:
                x_agent, o_agent = agent_b, agent_a
                a_is_x = False

            while not game.is_terminal():
                board = game.board
                player = game.current_player
                if player == 1:
                    action = x_agent(board, player)
                else:
                    action = o_agent(board, player)
                row, col = action // 3, action % 3
                game.step(row, col)

            status = game.status
            if status == 'X_win':
                a_won = a_is_x
                decisive = True
            elif status == 'O_win':
                a_won = not a_is_x
                decisive = True
            else:
                decisive = False

            if not decisive:
                draws += 1
            elif a_won:
                wins += 1
            else:
                losses += 1

        n = num_games if num_games > 0 else 1
        return {
            'wins': wins / n,
            'draws': draws / n,
            'losses': losses / n,
        }

    results = {}
    for agent_a, agent_b in matchups:
        key = f'{agent_a}_vs_{agent_b}'
        results[key] = play_matchup(agents[agent_a], agents[agent_b])

    return results
