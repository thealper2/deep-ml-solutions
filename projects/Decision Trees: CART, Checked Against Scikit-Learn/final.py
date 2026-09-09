"""
Decision Trees: CART, Checked Against Scikit-Learn — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  gini ──
import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

# ── Step 002  best_split ──
import numpy as np

def best_split(X, y):
    n = len(y)
    parent_gini = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0

    n_features = X.shape[1]

    for j in range(n_features):
        col = X[:, j]
        unique_vals = np.unique(col)

        if len(unique_vals) < 2:
            continue

        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for threshold in thresholds:
            left_mask = col <= threshold
            right_mask = ~left_mask
            n_left = np.sum(left_mask)
            n_right = n - n_left

            if n_left == 0 or n_right == 0:
                continue
            
            left_gini = gini(y[left_mask])
            right_gini = gini(y[right_mask])
            weighted_gini = (n_left * left_gini + n_right * right_gini) / n
            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = threshold

    if best_feature is None:
        return None, None, 0.0

    return best_feature, float(best_threshold), float(best_gain)

# ── Step 003  grow_tree ──
import numpy as np

def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    majority_class = int(classes[np.argmax(counts)])
    
    if len(classes) == 1 or depth >= max_depth:
        return {"leaf": True, "value": majority_class, "n": n}
    
    n_features = X.shape[1]
    parent_gini = gini(y)
    best_feature = None
    best_threshold = None
    best_gain = 0.0
    
    for j in range(n_features):
        col = X[:, j]
        unique_vals = np.unique(col)
        if len(unique_vals) < 2:
            continue
        
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        
        for threshold in thresholds:
            left_mask = col <= threshold
            right_mask = ~left_mask
            n_left = np.sum(left_mask)
            n_right = n - n_left
            
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            
            left_gini = gini(y[left_mask])
            right_gini = gini(y[right_mask])
            weighted_gini = (n_left * left_gini + n_right * right_gini) / n
            gain = parent_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = threshold
    
    if best_feature is None or best_gain <= 0:
        return {"leaf": True, "value": majority_class, "n": n}
    
    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    left_tree = grow_tree(X[left_mask], y[left_mask], max_depth, min_samples_leaf, depth + 1)
    right_tree = grow_tree(X[right_mask], y[right_mask], max_depth, min_samples_leaf, depth + 1)
    
    return {
        "leaf": False,
        "feature": best_feature,
        "threshold": best_threshold,
        "n": n,
        "left": left_tree,
        "right": right_tree
    }

# ── Step 004  predict_tree ──
import numpy as np

def predict_tree(tree, X):
    X = np.asarray(X)
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        node = tree
        while not node['leaf']:
            if X[i, node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
                
        predictions[i] = node['value']
    
    return predictions

# ── Step 005  fit_sklearn_tree ──
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

# ── Step 006  sklearn_splits ──
def sklearn_splits(clf):
    tree = clf.tree_
    splits = []

    for i in range(tree.node_count):
        if tree.children_left[i] != -1:
            feature = int(tree.feature[i])
            threshold = round(float(tree.threshold[i]), 3)
            splits.append((feature, threshold))

    return splits

# ── Step 007  compare_trees ──
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

# ── Step 008  moons_data ──
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# ── Step 009  overfit_vs_regularized ──
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

# ── Step 010  rotation_sensitivity ──
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

# ── Step 011  regression_tree ──
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def regression_tree(X, y, max_depth=2, random_state=42):
    reg = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    reg.fit(X, y)
    pred = reg.predict(X)
    n_distinct = len(np.unique(pred))
    mse = mean_squared_error(y, pred)

    return {
        "model": reg,
        "n_distinct_predictions": n_distinct,
        "train_mse": float(mse),
    }

# ── Step 012  tree_rules ──
from sklearn.tree import export_text

def tree_rules(clf, feature_names):
    return export_text(clf, feature_names=list(feature_names))

# ── Step 013  save_and_reload_tree ──
import joblib

def save_and_reload_tree(clf, path):
    joblib.dump(clf, path)
    return joblib.load(path)

# ── Step 014  predict_species ──
import numpy as np

def predict_species(clf, measurements, target_names):
    X = np.array(measurements)
    preds = clf.predict(X)
    return [str(target_names[int(p)]) for p in preds]

# ── Scaffold (runner) ──
"""Decision trees: CART by hand, checked against scikit-learn (Hands-On ML, chapter 5).

Story: grow a Gini tree on iris and confirm it matches DecisionTreeClassifier split
for split; watch a free tree memorize the moons data and min_samples_leaf fix it;
measure the axis-alignment bias with a rotation; fit a regression tree and count
its plateaus; then export the rules, save the model and serve species names.
"""
import os
import tempfile
import numpy as np
from sklearn.datasets import load_iris


def main() -> None:
    iris = load_iris()
    X2, y = iris.data[:, 2:], iris.target

    # ---- 1. Hand-grown CART vs scikit-learn ----
    print(f"iris root impurity {gini(y):.4f}; best split {best_split(X2, y)}")
    cmp = compare_trees(X2, y, max_depth=2)
    print(f"my depth-2 splits {cmp['my_splits']} | same as scikit-learn: {cmp['same_splits']} | "
          f"prediction agreement {cmp['agreement']:.3f}")
    print(tree_rules(fit_sklearn_tree(X2, y), iris.feature_names[2:]))

    # ---- 2. Overfitting and regularization on moons ----
    Xtr, Xte, ytr, yte = moons_data()
    r = overfit_vs_regularized(Xtr, Xte, ytr, yte, min_samples_leaf=5)
    print(f"moons free tree:        train {r['free']['train_acc']:.3f}  test {r['free']['test_acc']:.3f}  leaves {r['free']['leaves']}")
    print(f"moons min_samples_leaf=5: train {r['regularized']['train_acc']:.3f}  test {r['regularized']['test_acc']:.3f}  leaves {r['regularized']['leaves']}")
    rot = rotation_sensitivity(Xtr, Xte, ytr, yte, degrees=45.0)
    print(f"rotate the features 45 degrees: test accuracy {rot['original_acc']:.3f} -> {rot['rotated_acc']:.3f} (drop {rot['drop']:+.3f})")

    # ---- 3. Regression tree ----
    Xq = np.linspace(-1, 1, 200).reshape(-1, 1)
    yq = Xq.ravel() ** 2
    for d in (2, 3, 5):
        rr = regression_tree(Xq, yq, max_depth=d)
        print(f"regression tree depth {d}: {rr['n_distinct_predictions']:>2} distinct predictions, train MSE {rr['train_mse']:.4f}")

    # ---- 4. Ship ----
    final = fit_sklearn_tree(iris.data, iris.target, max_depth=3)
    path = os.path.join(tempfile.gettempdir(), "iris_tree.pkl")
    served = save_and_reload_tree(final, path)
    flowers = [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 4.2, 1.5], [6.7, 3.0, 5.2, 2.3]]
    print(f"\nserved: {predict_species(served, flowers, iris.target_names)} for {flowers}")


if __name__ == "__main__":
    main()
