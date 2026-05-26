import numpy as np
from collections import Counter

def calculate_oob_score(n_samples: int, bootstrap_indices: list, predictions: list, y_true: list) -> float:
    """
    Calculate the Out-of-Bag score for a bagging ensemble.
    
    Args:
        n_samples: Total number of samples in the dataset
        bootstrap_indices: List of lists containing indices used to train each estimator
        predictions: List of lists containing predictions from each estimator for all samples
        y_true: True labels for all samples
    
    Returns:
        OOB accuracy score as a float
    """
    n_estimators = len(bootstrap_indices)
    oob_predictions = [[] for _ in range(n_samples)]

    for i in range(n_estimators):
        in_bag = set(bootstrap_indices[i])
        for sample_idx in range(n_samples):
            if sample_idx not in in_bag:
                oob_predictions[sample_idx].append(predictions[i][sample_idx])

    correct = 0
    total = 0
    
    for sample_idx in range(n_samples):
        if len(oob_predictions[sample_idx]) == 0:
            continue
        
        counter = Counter(oob_predictions[sample_idx])
        majority_vote = counter.most_common(1)[0][0]

        if majority_vote == y_true[sample_idx]:
            correct += 1

        total += 1

    if total == 0:
        return 0.0

    return round(correct / total, 4)