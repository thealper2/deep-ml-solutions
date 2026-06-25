def train_q_learning_agent(num_episodes, alpha, gamma, initial_epsilon, min_epsilon, decay_rate, opponent_policy, rng):
    q_table = initialize_q_table()
    episode_outcomes = []
    agent_player = 1
    
    for episode in range(num_episodes):
        epsilon = epsilon_decay_schedule(initial_epsilon, episode, min_epsilon, decay_rate)
        
        game = TicTacToeGame()
        
        while not game.is_terminal():
            board = game.board
            current_player = game.current_player
            
            if current_player == agent_player:
                state_key, action = episode_agent_pick_action(q_table, board, current_player, epsilon, rng)
                out = episode_apply_action(board, action, current_player, agent_player)
                
                next_board = out['next_board']
                done = out['done']
                reward = out['reward']
                episode_apply_q_update(q_table, state_key, action, reward, next_board, done, alpha, gamma)
                
                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']
                
                if game.is_terminal():
                    break
            else:
                legal_moves = get_legal_moves(board)
                action = opponent_policy(board, current_player, rng)
                out = episode_apply_action(board, action, current_player, agent_player)
                
                next_board = out['next_board']
                done = out['done']
                reward = out['reward']
                state_key = canonical_board_key(board)
                episode_apply_q_update(q_table, state_key, action, reward, next_board, done, alpha, gamma)
                
                game.board = next_board
                game.current_player = out['next_player']
                game.status = out['status']
        
        episode_outcomes.append(game.status)
    
    return {'q_table': q_table, 'episode_outcomes': episode_outcomes}
