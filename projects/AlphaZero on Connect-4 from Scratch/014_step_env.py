def step_env(board, column, player):
    row = column_top_row(board, column)
    if row == -1:
        return board, False, 0, player
    
    new_board = board.copy()
    new_board[row, column] = player
    
    existing_winner = check_winner(board)
    
    done, winner = is_terminal(new_board)
    
    if done:
        if existing_winner != 0:
            next_player = existing_winner
        else:
            if winner != 0:
                next_player = 3 - player
            else:
                next_player = 0
    else:
        next_player = 3 - player
    
    return new_board, done, winner, next_player
