import numpy as np

def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0
) -> dict:
    """
    Evaluate LLM response using rubric-based multi-judge scoring.
    
    Args:
        judge_scores: 2D list where judge_scores[i][j] is judge i's score for criterion j
        criteria_weights: Weights for each criterion (should sum to 1)
        passing_threshold: Minimum normalized score to pass (0 to 1)
        max_score: Maximum possible score for each criterion
    
    Returns:
        Dictionary with evaluation results
    """
    judge_scores = np.array(judge_scores)
    num_judges, num_criteria = judge_scores.shape
    
    criterion_scores = np.mean(judge_scores, axis=0).tolist()
    weighted_score = np.sum(np.array(criterion_scores) * np.array(criteria_weights))
    normalized_score = weighted_score / max_score
    
    pass_status = normalized_score >= passing_threshold
    max_possible_std = max_score / 2
    std_per_criterion = np.std(judge_scores, axis=0, ddof=0)
    avg_std = np.mean(std_per_criterion)
    judge_agreement = max(0, min(1, 1 - (avg_std / max_possible_std)))
    
    return {
        'weighted_score': round(weighted_score, 4),
        'normalized_score': round(normalized_score, 4),
        'criterion_scores': [round(s, 4) for s in criterion_scores],
        'pass_status': pass_status,
        'judge_agreement': round(judge_agreement, 4)
    }
