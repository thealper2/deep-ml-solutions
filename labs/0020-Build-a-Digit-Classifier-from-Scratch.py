import math
import random

def train(X_train, y_train, X_val, y_val, n_classes):
    """
    Train a digit classifier using only Python and the math module.

    The dataset contains 8x8 pixel images of digits 0-9, flattened to 64 features.
    Pixel values are normalized to [0, 1].

    Args:
        X_train: list[list[float]] -- training images (each 64 floats)
        y_train: list[int]         -- training labels (each in [0, 10))
        X_val:   list[list[float]] -- validation images
        y_val:   list[int]         -- validation labels
        n_classes: int             -- number of classes (10)

    Returns:
        predict: callable, list[list[float]] -> list[int]
            Takes a list of images, returns a list of predicted class labels.
    """
    X_train = [list(x) for x in X_train]
    y_train = list(y_train)
    
    n_features = len(X_train[0])
    centroids = [[0.0] * n_features for _ in range(n_classes)]
    counts = [0] * n_classes
    
    for i, x in enumerate(X_train):
        label = y_train[i]
        for j in range(n_features):
            centroids[label][j] += x[j]
        counts[label] += 1
    
    for c in range(n_classes):
        if counts[c] > 0:
            for j in range(n_features):
                centroids[c][j] /= counts[c]
    
    total = len(X_train)
    class_weights = [1.0] * n_classes
    for c in range(n_classes):
        if counts[c] > 0:
            class_weights[c] = total / (n_classes * counts[c])
    
    feature_weights = [1.0] * n_features
    if len(X_val) > 0:
        all_data = X_train + [list(x) for x in X_val]
        means = [0.0] * n_features
        for x in all_data:
            for j in range(n_features):
                means[j] += x[j]
        for j in range(n_features):
            means[j] /= len(all_data)
        
        variances = [0.0] * n_features
        for x in all_data:
            for j in range(n_features):
                variances[j] += (x[j] - means[j]) ** 2
        for j in range(n_features):
            variances[j] /= len(all_data)
        
        max_var = max(variances) if max(variances) > 0 else 1.0
        feature_weights = [v / max_var + 0.1 for v in variances]
    
    def predict(X):
        results = []
        for x in X:
            scores = [0.0] * n_classes
            for c in range(n_classes):
                dist = 0.0
                for j in range(len(x)):
                    diff = x[j] - centroids[c][j]
                    dist += feature_weights[j] * diff * diff
                scores[c] = -dist * class_weights[c]
            
            best = 0
            best_score = scores[0]
            for c in range(1, n_classes):
                if scores[c] > best_score:
                    best_score = scores[c]
                    best = c
            results.append(best)
        return results
    
    best_accuracy = 0
    best_fw = feature_weights[:]
    
    if len(X_val) > 0:
        for scale in [0.5, 1.0, 1.5, 2.0]:
            test_fw = [w * scale for w in feature_weights]
            predictions = []
            for x in X_val:
                scores = [0.0] * n_classes
                for c in range(n_classes):
                    dist = 0.0
                    for j in range(len(x)):
                        diff = x[j] - centroids[c][j]
                        dist += test_fw[j] * diff * diff
                    scores[c] = -dist * class_weights[c]
                best = 0
                best_score = scores[0]
                for c in range(1, n_classes):
                    if scores[c] > best_score:
                        best_score = scores[c]
                        best = c
                predictions.append(best)
            
            correct = sum(1 for p, t in zip(predictions, y_val) if p == t)
            accuracy = correct / len(X_val)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_fw = test_fw[:]
        
        feature_weights = best_fw
    
    def final_predict(X):
        results = []
        for x in X:
            scores = [0.0] * n_classes
            for c in range(n_classes):
                dist = 0.0
                for j in range(len(x)):
                    diff = x[j] - centroids[c][j]
                    dist += feature_weights[j] * diff * diff
                scores[c] = -dist * class_weights[c]
            
            best = 0
            best_score = scores[0]
            for c in range(1, n_classes):
                if scores[c] > best_score:
                    best_score = scores[c]
                    best = c
            results.append(best)
        return results
    
    return final_predict
