def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    predictions = []
    for row in features:
        predictions.append(predict_example_tree(tree, row))
    
    return np.array(predictions)
