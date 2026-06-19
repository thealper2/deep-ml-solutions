def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    for row in range(5, -1, -1):
        if board[row, column] == 0:
            return row
    
    return -1
