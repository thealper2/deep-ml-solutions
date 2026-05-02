def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    result = []
    for candidate, score in zip(candidates, scores):
        max_score = max(score)
        output = candidate[score.index(max_score)]
        result.append(output)

    return result