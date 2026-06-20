def four_in_a_row_horizontal(board):
    for row in range(6):
        for col in range(4):
            if board[row, col]!= 0:
                if board[row, col] == board[row, col + 1] == board[row, col + 2] == board[row, col + 3]:
                    return int(board[row, col])

    return 0
