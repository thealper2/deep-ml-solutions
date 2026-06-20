def is_terminal(board):
    winner = check_winner(board)
    if winner != 0:
        return True, winner

    if board_is_full(board):
        return True, 0

    return False, 0
