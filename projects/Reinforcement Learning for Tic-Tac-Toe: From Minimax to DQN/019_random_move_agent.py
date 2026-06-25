import numpy as np

def random_move_agent(board, player, rng):
    """Return a uniformly random legal (row, col) move for `player`."""
    moves = get_legal_moves(board)
    idx = rng.integers(0, len(moves))
    return moves[idx]
