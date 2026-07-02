def predict_example_tree(tree, example):
    node = tree
    while not node['leaf']:
        feature_index = node['feature_index']
        threshold = node['threshold']
        if example[feature_index] <= threshold:
            node = node['left']
        else:
            node = node['right']

    return node['prediction']
