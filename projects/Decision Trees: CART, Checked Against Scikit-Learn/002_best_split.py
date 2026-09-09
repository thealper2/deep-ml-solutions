import numpy as np

def best_split(X, y):
    n = len(y)
    parent_gini = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    n_features = X.shape[1]

    for j in range(n_features):
        col = X[:, j]
        unique_vals = np.unique(col)

        if len(unique_vals) < 2:
            continue

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            left_mask = col <= threshold
            right_mask = ~left_mask
            n_left = np.sum(left_mask)
            n_right = n - n_left

            if n_left == 0 or n_right == 0:
                continue
            
            left_gini = gini(y[left_mask])
            right_gini = gini(y[right_mask])
            weighted_gini = (n_left * left_gini + n_right * right_gini) / n
            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = threshold

    if best_feature is None:
        return None, None, 0.0

    return best_feature, float(best_threshold), float(best_gain)