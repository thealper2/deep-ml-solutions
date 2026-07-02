def predict_forest(forest, features):
    tree_predictions = []
    for tree_info in forest:
        tree = tree_info['tree']
        tree_preds = predict_tree(tree, features)
        tree_predictions.append(tree_preds)

    tree_predictions = np.array(tree_predictions)
    return combine_predictions(tree_predictions)
