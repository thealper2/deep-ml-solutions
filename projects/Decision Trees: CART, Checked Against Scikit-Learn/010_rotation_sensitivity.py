import numpy as np

def rotation_sensitivity(X_train, X_test, y_train, y_test, degrees=45.0, min_samples_leaf=5):
    theta = np.deg2rad(degrees)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    X_train_rot = X_train @ R.T
    X_test_rot = X_test @ R.T
    
    tree_orig = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)
    orig_acc = accuracy_score(y_test, tree_orig.predict(X_test))
    
    tree_rot = fit_sklearn_tree(X_train_rot, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)
    rot_acc = accuracy_score(y_test, tree_rot.predict(X_test_rot))
    
    drop = orig_acc - rot_acc
    
    return {
        "original_acc": float(orig_acc),
        "rotated_acc": float(rot_acc),
        "drop": round(float(drop), 4)
    }