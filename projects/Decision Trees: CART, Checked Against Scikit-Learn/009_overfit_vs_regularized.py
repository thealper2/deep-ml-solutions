from sklearn.metrics import accuracy_score

def overfit_vs_regularized(X_train, X_test, y_train, y_test, min_samples_leaf=5):
    free_tree = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=1)
    free_train_acc = accuracy_score(y_train, free_tree.predict(X_train))
    free_test_acc = accuracy_score(y_test, free_tree.predict(X_test))
    free_leaves = free_tree.get_n_leaves()
    
    reg_tree = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)
    reg_train_acc = accuracy_score(y_train, reg_tree.predict(X_train))
    reg_test_acc = accuracy_score(y_test, reg_tree.predict(X_test))
    reg_leaves = reg_tree.get_n_leaves()
    
    return {
        "free": {
            "train_acc": float(free_train_acc),
            "test_acc": float(free_test_acc),
            "leaves": int(free_leaves)
        },
        "regularized": {
            "train_acc": float(reg_train_acc),
            "test_acc": float(reg_test_acc),
            "leaves": int(reg_leaves)
        }
    }