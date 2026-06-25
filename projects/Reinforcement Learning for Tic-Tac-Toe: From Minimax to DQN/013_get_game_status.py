import numpy as np

def get_game_status(board):
    """Return 'X_win', 'O_win', 'draw', or 'ongoing' for the given 3x3 board."""
    if is_winner(board, 1):
        return 'X_win'
    if is_winner(board, -1):
        return 'O_win'
    if is_draw(board):
        return 'draw'
    
    return 'ongoing' 
