def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    n_features = features.shape[1]
    if feature_subset is None:
        available_features = list(range(n_features))
    else:
        available_features = list(feature_subset) if not isinstance(feature_subset, list) else feature_subset
    
    split = best_split(features, labels, available_features)
    
    if split['score'] <= 0 or split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    feature_idx = split['feature_index']
    threshold = split['threshold']
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feature_idx, threshold)
    
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    left_tree = build_tree(left_X, left_y, max_depth, min_samples_split, feature_subset, depth + 1)
    right_tree = build_tree(right_X, right_y, max_depth, min_samples_split, feature_subset, depth + 1)
    
    return {
        'leaf': False,
        'feature_index': feature_idx,
        'threshold': threshold,
        'left': left_tree,
        'right': right_tree
    }
