import numpy as np

def play_hardcoded_game(moves):
    """Replay a fixed sequence of (row, col) moves and return (final_board, status)."""
    board = np.zeros((3, 3), dtype=int)
    player = 1
    status = 'ongoing'

    for row, col in moves:
        try:
            board = place_move(board, row, col, player)
        except ValueError:
            break

        status = get_game_status(board)
        if status != 'ongoing':
            break

        player = switch_player(player)

    return board, status
