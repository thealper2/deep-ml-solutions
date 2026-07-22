import numpy as np

def multi_hypothesis_trajectory_eval(
    predictions: list,
    probabilities: list,
    ground_truth: list
) -> dict:
    """
    Evaluate multi-hypothesis trajectory predictions against ground truth.

    Args:
        predictions: K predicted trajectories, each of shape [T, 2]
        probabilities: K probability values for each hypothesis
        ground_truth: Single ground truth trajectory of shape [T, 2]

    Returns:
        Dictionary with evaluation metrics
    """
    K = len(predictions)
    T = len(ground_truth)

    ade_list = []
    fde_list = []

    for pred in predictions:
        pred = np.array(pred)
        gt = np.array(ground_truth)

        distances =np.sqrt(np.sum((pred - gt) ** 2, axis=1))
        
        ade = np.mean(distances)
        fde = distances[-1]

        ade_list.append(float(ade))
        fde_list.append(float(fde))

    min_ade = min(ade_list)
    min_fde = min(fde_list)
    best_idx = int(np.argmin(ade_list))

    wta_loss = min_ade
    prob_weighted_ade = sum(p * ade for p, ade in zip(probabilities, ade_list))

    return {
        'ade_per_hypothesis': [round(v, 4) for v in ade_list],
        'fde_per_hypothesis': [round(v, 4) for v in fde_list],
        'min_ade': round(min_ade, 4),
        'min_fde': round(min_fde, 4),
        'best_hypothesis_idx': best_idx,
        'wta_loss': round(wta_loss, 4),
        'prob_weighted_ade': round(prob_weighted_ade, 4),
    }
