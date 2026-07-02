import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    n, d = features.shape
    rng = np.random.default_rng(random_state)
    
    if num_features_per_split is None:
        num_features_per_split = max(1, int(np.sqrt(d)))
    
    forest = []
    for _ in range(num_trees):
        X_boot, y_boot = bootstrap_sample(features, labels, rng)
        feat_indices = feature_subset(d, num_features_per_split, rng)
        tree = build_tree(X_boot, y_boot, max_depth, min_samples_split, list(feat_indices))
        
        forest.append({
            'tree': tree,
            'feature_indices': feat_indices
        })
    
    return forest
