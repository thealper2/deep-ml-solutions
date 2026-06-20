def four_in_a_row_vertical(board):
    for col in range(7):
        for row in range(3):
            if board[row, col] != 0:
                if board[row, col] == board[row + 1, col] == board[row + 2, col] == board[row + 3, col]:
                    return int(board[row, col])

    return 0
