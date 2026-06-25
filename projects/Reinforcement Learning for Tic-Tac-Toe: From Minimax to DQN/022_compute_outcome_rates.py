def compute_outcome_rates(outcomes):
    """Return {'x_win_rate','o_win_rate','draw_rate'} from a list of outcome labels."""
    n = len(outcomes)
    if not outcomes:
        return {'x_win_rate': 0.0, 'o_win_rate': 0.0, 'draw_rate': 0.0}
    
    x_win_rate = outcomes.count('X_win') / n
    o_win_rate = outcomes.count('O_win') / n
    if x_win_rate == 0.0 and o_win_rate == 0.0:
        draw_rate = 1.0
    else:
        draw_rate = outcomes.count('draw') / n
    return {'x_win_rate': x_win_rate, 'o_win_rate': o_win_rate, 'draw_rate': draw_rate}
