def random_forest_feature_importance(trees: list, n_features: int) -> list:
    """
    Calculate feature importance from a random forest using Mean Decrease in Impurity.
    
    Args:
        trees: List of trees, where each tree is a list of node splits.
               Each split is a dict with:
               - 'feature_index': int, the feature used for splitting
               - 'impurity_decrease': float, the weighted impurity decrease
        n_features: Total number of features in the dataset
    
    Returns:
        List of feature importances normalized to sum to 1.0
    """
    features = {i: 0 for i in range(n_features)}
    for tree in trees:
        for feature in tree:
            idx = feature['feature_index']
            impurity_decrease = feature['impurity_decrease']
            features[idx] = features.get(idx, 0) + impurity_decrease

    result = list(features.values())[:n_features]
    return result