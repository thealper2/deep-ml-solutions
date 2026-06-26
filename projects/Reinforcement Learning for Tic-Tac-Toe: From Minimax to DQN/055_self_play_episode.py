def self_play_episode(q_table, alpha, gamma, epsilon, rng):
    """Run one self-play episode and return final_status and a list of transitions."""
    board, player = episode_reset_game()
    transitions = []
    
    while True:
        state_key, action = episode_agent_pick_action(q_table, board, player, epsilon, rng)
        out = episode_apply_action(board, action, player, player)
        
        transitions.append({
            'state_key': state_key,
            'action': action,
            'reward': out['reward'],
            'next_board': out['next_board'],
            'done': out['done'],
            'player': player
        })
        
        board = out['next_board']
        if out['done']:
            final_status = out['status']
            break
        
        player = out['next_player']
    
    return {
        'final_status': final_status,
        'transitions': transitions
    }
