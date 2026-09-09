def compare_trees(X, y, max_depth=2):
    my_tree = grow_tree(X, y, max_depth=max_depth, min_samples_leaf=1)
    sklearn_tree = fit_sklearn_tree(X, y, max_depth=max_depth, min_samples_leaf=1)
    my_splits = extract_splits(my_tree)
    sk_splits = sklearn_splits(sklearn_tree)
    same_splits = my_splits == sk_splits
    my_preds = predict_tree(my_tree, X)
    sk_preds = sklearn_tree.predict(X)
    agreement = np.mean(my_preds == sk_preds)
    
    return {
        "same_splits": same_splits,
        "agreement": float(agreement),
        "my_splits": my_splits
    }

def extract_splits(tree):
    splits = []
    def traverse(node):
        if not node['leaf']:
            splits.append((node['feature'], round(float(node['threshold']), 3)))
            traverse(node['left'])
            traverse(node['right'])

    traverse(tree)
    return splits