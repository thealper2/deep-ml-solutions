def minimax_terminal_score(status):
    """Return +1 for 'X_win', -1 for 'O_win', 0 for 'draw'."""
    d = {'X_win': 1, 'O_win': -1, 'draw': 0}
    return d[status]
