def score_dialog_complexity(dialogs):
    """
    Args:
        dialogs: list of lists of intention tag strings
    Returns:
        list of (index, score) tuples sorted by score desc, then index asc
    """
    scores = []
    for i, dialog in enumerate(dialogs):
        unique_tags = len(set(dialog))
        scores.append((i, unique_tags))

    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores