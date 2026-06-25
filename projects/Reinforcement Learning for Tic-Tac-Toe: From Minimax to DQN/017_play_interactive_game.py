def play_interactive_game():
    """Play a full game with two humans entering moves via stdin and return the final status."""
    board = np.zeros((3, 3), dtype=int)
    player = 1
    status = 'ongoing'

    print_board(board)

    while get_legal_moves(board) != []:
        try:
            row, col = map(int, input().split())
            board = place_move(board, row, col, player)
        except ValueError:
            print_board(board)
            continue

        print_board(board)

        status = get_game_status(board)
        if status != 'ongoing':
            break

        player = switch_player(player)

    return status
