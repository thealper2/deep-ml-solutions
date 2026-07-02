"""
Random Forest from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  impurity ──
import numpy as np

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    if len(labels) == 0:
        return 0.0

    counts = np.bincount(labels)
    probs = counts / len(labels)
    return 1.0 - np.sum(probs ** 2)

# ── Step 002  split_dataset ──
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    left_indices = features[:, feature_index] <= threshold
    right_indices = ~left_indices
    
    return features[left_indices], labels[left_indices], features[right_indices], labels[right_indices]

# ── Step 003  split_score ──
def split_score(parent_labels, left_labels, right_labels):
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    
    parent_impurity = impurity(parent_labels)
    weighted_child_impurity = (n_left / n) * impurity(left_labels) + (n_right / n) * impurity(right_labels)
    
    return parent_impurity - weighted_child_impurity

# ── Step 004  best_split ──
import numpy as np

def best_split(features, labels, feature_indices):
    best_score = 0.0
    best_feature = None
    best_threshold = None
    n = len(features)

    for feature_idx in feature_indices:
        col = features[:, feature_idx]
        unique_values = np.unique(col)

        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2

            left_mask = col <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            left_labels = labels[left_mask]
            right_labels = labels[right_mask]

            score = split_score(labels, left_labels, right_labels)

            if score > best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = threshold

    return {
        'feature_index': best_feature,
        'threshold': best_threshold,
        'score': best_score,
    }

# ── Step 005  should_stop ──
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if len(np.unique(labels)) == 1:
        return True

    if depth >= max_depth:
        return True

    if len(labels) < min_samples_split:
        return True

    return False

# ── Step 006  leaf_prediction ──
def leaf_prediction(labels):
    counts = np.bincount(labels)
    return int(np.argmax(counts))

# ── Step 007  build_tree ──
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    n_features = features.shape[1]
    if feature_subset is None:
        available_features = list(range(n_features))
    else:
        available_features = list(feature_subset) if not isinstance(feature_subset, list) else feature_subset
    
    split = best_split(features, labels, available_features)
    
    if split['score'] <= 0 or split['feature_index'] is None:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    feature_idx = split['feature_index']
    threshold = split['threshold']
    left_X, left_y, right_X, right_y = split_dataset(features, labels, feature_idx, threshold)
    
    if len(left_y) == 0 or len(right_y) == 0:
        return {'leaf': True, 'prediction': leaf_prediction(labels)}
    
    left_tree = build_tree(left_X, left_y, max_depth, min_samples_split, feature_subset, depth + 1)
    right_tree = build_tree(right_X, right_y, max_depth, min_samples_split, feature_subset, depth + 1)
    
    return {
        'leaf': False,
        'feature_index': feature_idx,
        'threshold': threshold,
        'left': left_tree,
        'right': right_tree
    }

# ── Step 008  predict_example_tree ──
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

# ── Step 009  predict_tree ──
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

# ── Step 010  bootstrap_sample ──
def bootstrap_sample(features, labels, rng):
    n = len(features)
    indices = rng.integers(0, n, size=n)
    return features[indices], labels[indices]

# ── Step 011  feature_subset ──
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    return rng.choice(num_features, size=num_to_pick, replace=False)

# ── Step 012  train_forest ──
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    n, d = features.shape
    rng = np.random.default_rng(random_state)
    
    if num_features_per_split is None:
        num_features_per_split = max(1, int(np.sqrt(d)))
    
    forest = []
    for _ in range(num_trees):
        X_boot, y_boot = bootstrap_sample(features, labels, rng)
        feat_indices = feature_subset(d, num_features_per_split, rng)
        tree = build_tree(X_boot, y_boot, max_depth, min_samples_split, list(feat_indices))
        
        forest.append({
            'tree': tree,
            'feature_indices': feat_indices
        })
    
    return forest

# ── Step 013  combine_predictions ──
def combine_predictions(tree_predictions):
    n_trees, n_samples = tree_predictions.shape
    result = np.zeros(n_samples, dtype=int)

    for j in range(n_samples):
        votes = tree_predictions[:, j]
        counts = np.bincount(votes)
        result[j] = np.argmax(counts)

    return result

# ── Step 014  predict_forest ──
def predict_forest(forest, features):
    tree_predictions = []
    for tree_info in forest:
        tree = tree_info['tree']
        tree_preds = predict_tree(tree, features)
        tree_predictions.append(tree_preds)

    tree_predictions = np.array(tree_predictions)
    return combine_predictions(tree_predictions)

# ── Step 015  accuracy ──
def accuracy(predictions, labels):
    return (predictions == labels).sum() / len(labels)

# ── Scaffold (runner) ──
"""Scaffold for: Random Forest from Scratch.

Build a small synthetic dataset, train a random forest, predict, and report
train/test accuracy. Every function is concatenated above this scaffold, so
they are called directly (there is no separate solution module).
"""

import numpy as np


def main():
    rng = np.random.default_rng(0)
    n, d = 150, 5
    X = rng.random((n, d))
    y = ((X[:, 0] + X[:, 3]) > 1.0).astype(int)
    split = 110
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    forest = train_forest(X_train, y_train, num_trees=15, max_depth=6, random_state=0)
    print("forest size:", len(forest))

    train_acc = accuracy(predict_forest(forest, X_train), y_train)
    test_acc = accuracy(predict_forest(forest, X_test), y_test)
    print("train accuracy:", round(float(train_acc), 4))
    print("test accuracy: ", round(float(test_acc), 4))

    single = build_tree(X_train, y_train, max_depth=6)
    print("single-tree train accuracy:", round(float(accuracy(predict_tree(single, X_train), y_train)), 4))


if __name__ == "__main__":
    main()
