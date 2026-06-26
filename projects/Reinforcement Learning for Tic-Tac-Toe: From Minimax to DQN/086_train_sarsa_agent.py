def train_sarsa_agent(num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, opponent_policy, rng):
    q_table = initialize_q_table()
    episode_outcomes = []
    agent_player = 1

    for episode in range(num_episodes):
        epsilon = epsilon_decay_schedule(initial_epsilon, episode, min_epsilon, decay_rate)
        
        game = TicTacToeGame()
        prev_state_key = None
        prev_action = None
        prev_reward = None

        while not game.is_terminal():
            board = game.board
            current_player = game.current_player

            if current_player == agent_player:
                state_key, action = episode_agent_pick_action(q_table, board, current_player, epsilon, rng)

                if prev_state_key is not None:
                    done = False
                    next_state_key = state_key
                    q_table = sarsa_on_policy_update(
                        q_table, prev_state_key, prev_action, prev_reward,
                        next_state_key, action, False, alpha, gamma
                    )

                out = episode_apply_action(board, action, current_player, agent_player)

                next_board = out['next_board']
                done = out['done']
                reward = out['reward']

                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']

                if done:
                    if prev_state_key is not None:
                        q_table = sarsa_on_policy_update(
                            q_table, prev_state_key, prev_action, prev_reward,
                            state_key, action, True, alpha, gamma
                        )

                    q_table = sarsa_on_policy_update(
                        q_table, state_key, action, reward,
                        None, None, True, alpha, gamma
                    )
                    break

                prev_state_key = state_key
                prev_action = action
                prev_reward = reward
            
            else:
                legal_moves = get_legal_moves(board)
                action = opponent_policy(board, current_player, rng)
                out = episode_apply_action(board, action, current_player, agent_player)

                next_board = out['next_board']
                done = out['done']
                reward = out['reward']

                if prev_state_key is not None and not done:
                    q_table = sarsa_on_policy_update(
                        q_table, prev_state_key, prev_action, prev_reward,
                        canonical_board_key(board), action, False, alpha, gamma
                    )
                    prev_state_key = None
                    prev_action = None
                    prev_reward = None

                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']

        episode_outcomes.append(game.status)

    return {'q_table': q_table, 'episode_outcomes': episode_outcomes}
