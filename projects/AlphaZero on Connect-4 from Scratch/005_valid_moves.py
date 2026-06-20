def valid_moves(board):
    columns = board.shape[1]
    valid = [column for column in range(columns) if not column_full(board, column)]
    return valid
