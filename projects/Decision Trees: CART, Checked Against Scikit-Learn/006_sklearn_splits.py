def sklearn_splits(clf):
    tree = clf.tree_
    splits = []

    for i in range(tree.node_count):
        if tree.children_left[i] != -1:
            feature = int(tree.feature[i])
            threshold = round(float(tree.threshold[i]), 3)
            splits.append((feature, threshold))

    return splits
