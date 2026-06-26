import numpy as np

def inspect_q_values_for_state(q_table, board, current_player):
    """Print the board and Q-values for all 9 cells; return a length-9 array."""
    print_board(board)

    state_key = canonical_board_key(board)

    values = np.zeros(9)
    for i in range(9):
        row = i // 3
        col = i % 3
        values[i] = get_q_value(q_table, state_key, (row, col))

    for row in range(3):
        line = []
        for  col in range(3):
            idx = row * 3 + col
            line.append(f"{values[idx]:+.2f}")

        print(' '.join(line))

    return values
