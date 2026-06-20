def board_is_full(board):
    for col in range(board.shape[1]):
        if board[0, col] == 0:
            return False
            
    return True
