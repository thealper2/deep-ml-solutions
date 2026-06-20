def four_in_a_row_diagonal_up_right(board):
    for row in range(3, 6):
        for col in range(4):
            if board[row, col] != 0:
                if board[row, col] == board[row - 1, col + 1] == board[row - 2, col + 2] == board[row - 3, col + 3]:
                    return int(board[row, col])
    return 0
