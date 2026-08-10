import numpy as np


class DecisionTree:
    """
    A decision tree classifier that the harness will use as the base learner
    in a Random Forest. The harness will create many DecisionTree instances,
    train each on a different bootstrap sample of the data, and aggregate
    their predictions by majority vote.

    For the ensemble to beat a single tree by a meaningful margin, your trees
    must be DIVERSE. Bootstrap sampling (handled by the harness) gives some
    diversity. The most effective additional source of diversity is to
    randomize WHICH features each split considers -- this is what makes a
    Random Forest different from plain Bagging.

    Use `self.random_state` for any randomness inside your tree so that
    different harness seeds produce different trees.
    """

    def __init__(self, max_depth=10, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def fit(self, X, y):
        """
        Fit the tree on (X, y).

        Args:
            X: numpy array of shape (n_samples, n_features), dtype float
            y: numpy array of shape (n_samples,), integer class labels in [0, n_classes)

        Returns:
            self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        
        if self.max_features == 'sqrt':
            self.n_features_split = int(np.sqrt(self.n_features))
        elif self.max_features is None:
            self.n_features_split = self.n_features
        else:
            self.n_features_split = min(self.max_features, self.n_features)
        
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_classes = self.n_classes
        
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return self._make_leaf(y)
        
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            return self._make_leaf(y)
        
        best_feature, best_threshold, best_gini = self._best_split(X, y)
        
        if best_feature is None:
            return self._make_leaf(y)
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self._make_leaf(y)
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child,
            'is_leaf': False
        }

    def _best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        n_samples = len(y)
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        features = np.random.choice(
            self.n_features, 
            size=self.n_features_split, 
            replace=False
        )
        
        for feature in features:
            values = X[:, feature]
            unique_values = np.unique(values)
            
            if len(unique_values) <= 1:
                continue
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_mask = values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gini = self._gini_impurity(y[left_mask], y[right_mask])
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def _gini_impurity(self, left_y, right_y):
        """Compute weighted Gini impurity for a split."""
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right
        
        def gini(y):
            if len(y) == 0:
                return 0
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1 - np.sum(probs ** 2)
        
        return (n_left / n_total) * gini(left_y) + (n_right / n_total) * gini(right_y)
    
    def _make_leaf(self, y):
        """Create a leaf node with the most common label."""
        counts = np.bincount(y, minlength=self.n_classes)
        return {
            'prediction': np.argmax(counts),
            'counts': counts,
            'is_leaf': True
        }

    def predict(self, X):
        """
        Predict integer class labels for X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples,) with integer class labels
        """
        if self.tree is None:
            raise ValueError("Tree not fitted yet")
        
        predictions = np.array([self._predict_single(x, self.tree) for x in X])
        return predictions
    
    def _predict_single(self, x, node):
        """Recursively traverse tree for a single sample."""
        if node['is_leaf']:
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
