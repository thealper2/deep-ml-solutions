import numpy as np

def best_split(features, labels, feature_indices):
    best_score = 0.0
    best_feature = None
    best_threshold = None
    n = len(features)

    for feature_idx in feature_indices:
        col = features[:, feature_idx]
        unique_values = np.unique(col)

        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2

            left_mask = col <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            score = split_score(labels, left_labels, right_labels)

            if score > best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = threshold

    return {
        'feature_index': best_feature,
        'threshold': best_threshold,
        'score': best_score,
    }
