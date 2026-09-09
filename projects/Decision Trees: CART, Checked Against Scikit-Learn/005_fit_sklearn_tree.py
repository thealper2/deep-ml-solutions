from sklearn.tree import DecisionTreeClassifier

def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        criterion='gini'
    )
    tree.fit(X, y)
    return tree