def ply_based_discount(evaluations: list, ply_depths: list, gamma: float) -> tuple:
    """
    Apply ply-based discounting to terminal evaluations at various search depths.
    
    Args:
        evaluations: Terminal evaluation scores for each candidate move sequence
        ply_depths: Number of plies to reach each terminal evaluation
        gamma: Per-ply discount factor (0 < gamma <= 1)
    
    Returns:
        Tuple of (discounted_values, best_index)
    """
    values = []
    for evaluation, ply_depth in zip(evaluations, ply_depths):
        value = evaluation * (gamma ** ply_depth)
        values.append(value)

    return [round(v, 4) for v in values], values.index(max(values))