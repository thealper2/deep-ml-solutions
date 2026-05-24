import numpy as np
from typing import Tuple

def gini_impurity(y):
    if len(y) == 0:
        return 0

    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    n_samples = len(y)

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            gini_left = gini_impurity(y[left_mask])
            gini_right = gini_impurity(y[right_mask])

            n_l, n_r = np.sum(left_mask), np.sum(right_mask)
            weighted_gini = (n_l / n_samples) * gini_left + (n_r / n_samples) * gini_right

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold