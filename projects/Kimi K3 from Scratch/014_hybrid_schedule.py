def hybrid_schedule(n_repeats):
    """['KDA','KDA','KDA','MLA'] repeated n_repeats times, plus a final 'MLA'."""
    pattern = ['KDA', 'KDA', 'KDA', 'MLA']
    return pattern * n_repeats + ['MLA']
