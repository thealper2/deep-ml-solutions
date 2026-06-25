def compute_outcome_rates(outcomes):
    """Return {'x_win_rate','o_win_rate','draw_rate'} from a list of outcome labels."""
    n = len(outcomes)
    x_win_rate = outcomes.count('X_win') / n if n > 0 else 0.0
    o_win_rate = outcomes.count('O_win') / n if n > 0 else 0.0
    draw_rate = outcomes.count('draw') / n if n > 0 else 0.0
    return {'x_win_rate': x_win_rate, 'o_win_rate': o_win_rate, 'draw_rate': draw_rate}
