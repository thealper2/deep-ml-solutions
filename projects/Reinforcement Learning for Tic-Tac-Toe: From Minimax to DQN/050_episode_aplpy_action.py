def episode_apply_action(board, action, current_player, agent_player):
    """Apply one move, return next_board/next_player/status/reward/done."""
    row = action // 3
    col = action % 3
    
    new_board = place_move(board, row, col, current_player)
    status = get_game_status(new_board)
    
    reward = tic_tac_toe_reward(status, agent_player)
    done = (status != 'ongoing')
    
    next_player = switch_player(current_player)
    
    return {
        'next_board': new_board,
        'next_player': next_player,
        'status': status,
        'reward': reward,
        'done': done
    }
