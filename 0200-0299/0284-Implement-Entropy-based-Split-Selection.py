import numpy as np

def entropy(labels):
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-12))

def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    n_samples, n_features = X.shape
    best_feature = None
    best_threshold = None
    best_info_gain = -1.0

    parent_entropy = entropy(y)

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_vals = np.unique(feature_values)

        if len(unique_vals) == 1:
            continue

        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2

            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            left_labels = y[left_mask]
            right_labels = y[right_mask]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            
            left_entropy = entropy(left_labels)
            right_entropy = entropy(right_labels)

            weighted_entropy = (len(left_labels) / n_samples) * left_entropy + (len(right_labels) / n_samples) * right_entropy

            info_gain = parent_entropy - weighted_entropy

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_info_gain
