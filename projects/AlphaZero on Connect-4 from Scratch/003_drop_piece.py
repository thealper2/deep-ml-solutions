def drop_piece(board, column, player):
    new_board = board.copy()
    row = column_top_row(new_board, column)
    if row == -1:
        raise ValueError('')

    new_board[row, column] = player
    return new_board
