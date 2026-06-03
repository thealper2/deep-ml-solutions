def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    """
    Analyze pairwise comparisons between LLM responses.
    
    Args:
        comparisons: List of comparison dicts with 'id', 'scores_a', 'scores_b'
        criteria_weights: Dict mapping criterion names to importance weights
        tie_threshold: Maximum difference to declare a tie
    
    Returns:
        Dict with 'results', 'win_rate_a', 'win_rate_b', 'tie_rate', 'avg_margin'
    """
    if not comparisons:
        return {
            'results': [],
            'win_rate_a': 0.0,
            'win_rate_b': 0.0,
            'tie_rate': 0.0,
            'avg_margin': 0.0,
        }

    total_weight = sum(criteria_weights.values())
    norm_weights = {k: v / total_weight for k, v in criteria_weights.items()}

    results = []
    wins_a = 0
    wins_b = 0
    ties = 0
    total_margin = 0.0

    for comp in comparisons:
        score_a = sum(norm_weights[c] * comp['scores_a'][c] for c in norm_weights)
        score_b = sum(norm_weights[c] * comp['scores_b'][c] for c in norm_weights)

        margin = abs(score_a - score_b)
        total_margin += margin

        if margin <= tie_threshold:
            winner = 'tie'
            ties += 1
        elif score_a > score_b:
            winner = 'A'
            wins_a += 1
        else:
            winner = 'B'
            wins_b += 1

        results.append({
            'id': comp['id'],
            'winner': winner,
            'margin': round(margin, 4),
        })

    n = len(comparisons)
    win_rate_a = wins_a / n
    win_rate_b = wins_b / n
    tie_rate = ties / n
    avg_margin = total_margin / n

    return {
        'results': results,
        'win_rate_a': round(win_rate_a, 4),
        'win_rate_b': round(win_rate_b, 4),
        'tie_rate': round(tie_rate, 4),
        'avg_margin': round(avg_margin, 4),
    }