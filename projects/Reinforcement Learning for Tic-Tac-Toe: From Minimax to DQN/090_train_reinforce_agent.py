def train_reinforce_agent(num_episodes, gamma, learning_rate, hidden_dim, opponent_policy, rng, init_seed=0):
    arch = build_mlp_architecture(9, hidden_dim, 9)
    params = initialize_mlp_parameters(arch, seed=init_seed)
    adam_state = {}
    episode_outcomes = []
    agent_player = 1
    
    for episode in range(num_episodes):
        game = TicTacToeGame()
        states = []
        actions = []
        legal_masks = []
        rewards = []
        
        while not game.is_terminal():
            board = game.board
            current_player = game.current_player
            
            if current_player == agent_player:
                state = encode_board_flat_length_nine(board, current_player)
                legal_mask = np.zeros(9, dtype=bool)
                for row, col in get_legal_moves(board):
                    legal_mask[row * 3 + col] = True
                
                q_values, _ = mlp_forward_pass(params, state.reshape(1, -1))
                masked_q = mask_illegal_actions_neg_inf(q_values.flatten(), legal_mask)
                
                max_logit = np.max(masked_q)
                exp_logits = np.exp(masked_q - max_logit)
                probs = exp_logits / np.sum(exp_logits)
                
                action = rng.choice(9, p=probs)
                
                states.append(state)
                actions.append(action)
                legal_masks.append(legal_mask.astype(bool))
                
                out = episode_apply_action(board, action, current_player, agent_player)
                reward = out['reward']
                rewards.append(reward)
                
                game.board = out['next_board']
                game.current_player = out['next_player']
                game.status = out['status']
            else:
                move = opponent_policy(board, current_player, rng)
                if isinstance(move, tuple):
                    row, col = move
                else:
                    row = move // 3
                    col = move % 3
                out = episode_apply_action(board, row * 3 + col, current_player, agent_player)
                
                game.board = out['next_board']
                game.current_player = out['next_player']
                game.status = out['status']
        
        if len(rewards) > 0:
            returns = reinforce_collect_episode_returns(rewards, gamma)
            episode_cache = {
                'states': np.array(states),
                'actions': np.array(actions),
                'legal_masks': np.array(legal_masks)
            }
            params, adam_state = reinforce_policy_gradient_update(
                params, episode_cache, returns, adam_state, learning_rate
            )
        
        episode_outcomes.append(game.status)
    
    return {
        'params': params,
        'architecture': arch,
        'episode_outcomes': episode_outcomes
    }
