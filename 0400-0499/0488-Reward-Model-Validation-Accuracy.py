import numpy as np

def reward_model_validation(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    margin_thresholds: list[float]
) -> dict:
    """
    Compute validation metrics for a reward model on held-out preference pairs.

    Args:
        chosen_scores: 1D array of reward scores for preferred responses.
        rejected_scores: 1D array of reward scores for rejected responses.
        margin_thresholds: List of thresholds for margin-based accuracy.

    Returns:
        Dictionary with 'accuracy', 'mean_margin', 'concordance', 'margin_accuracy'.
    """
    margins = chosen_scores - rejected_scores
    n = len(margins)
    wins = sum(margins > 0)
    ties = sum(margins == 0)
    accuracy = wins / n
    mean_margin = sum(margins) / n
    concordance = (wins + 0.5 * ties) / n
    margin_accuracy = {threshold: round(sum(margins >= threshold) / n, 4) for threshold in margin_thresholds}
    return {
        'accuracy': round(accuracy, 4),
        'mean_margin': round(mean_margin, 4),
        'concordance': round(concordance, 4),
        'margin_accuracy': margin_accuracy,
    }
