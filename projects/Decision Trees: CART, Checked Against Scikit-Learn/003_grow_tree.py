import numpy as np

def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    majority_class = int(classes[np.argmax(counts)])
    
    if len(classes) == 1 or depth >= max_depth:
        return {"leaf": True, "value": majority_class, "n": n}
    
    n_features = X.shape[1]
    parent_gini = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0
    
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
            
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            
            left_gini = gini(y[left_mask])
            right_gini = gini(y[right_mask])
            weighted_gini = (n_left * left_gini + n_right * right_gini) / n
            gain = parent_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = threshold
    
    if best_feature is None or best_gain <= 0:
        return {"leaf": True, "value": majority_class, "n": n}
    
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    left_tree = grow_tree(X[left_mask], y[left_mask], max_depth, min_samples_leaf, depth + 1)
    right_tree = grow_tree(X[right_mask], y[right_mask], max_depth, min_samples_leaf, depth + 1)
    
    return {
        "leaf": False,
        "feature": best_feature,
        "threshold": best_threshold,
        "n": n,
        "left": left_tree,
        "right": right_tree
    }