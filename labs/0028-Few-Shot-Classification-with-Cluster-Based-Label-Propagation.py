import numpy as np

def train(X_unlabeled, X_labeled, y_labeled, X_val, y_val):
    """
    Train a classifier in a semi-supervised setting using cluster-based label propagation.
    Pure numpy implementation - no sklearn.
    """
    X_all = np.vstack([X_labeled, X_unlabeled])
    n_labeled = X_labeled.shape[0]
    n_samples = X_all.shape[0]
    
    mean = np.mean(X_all, axis=0)
    std = np.std(X_all, axis=0)
    std[std == 0] = 1.0
    X_all_norm = (X_all - mean) / std
    
    def kmeans(X, k, max_iters=100, seed=42):
        np.random.seed(seed)
        n, d = X.shape
        centroids = np.zeros((k, d))
        centroids[0] = X[np.random.randint(n)]
        for i in range(1, k):
            dists = np.min([np.sum((X - c)**2, axis=1) for c in centroids[:i]], axis=0)
            probs = dists / np.sum(dists)
            centroids[i] = X[np.random.choice(n, p=probs)]
        
        for _ in range(max_iters):
            dists = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        return labels, centroids
    
    def knn_predict(X_train, y_train, X_test, k=3):
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        for i in range(n_test):
            dists = np.sum((X_train - X_test[i])**2, axis=1)
            nearest = np.argsort(dists)[:k]
            labels = y_train[nearest]
            counts = np.bincount(labels, minlength=10)
            predictions[i] = np.argmax(counts)
        return predictions
    
    def logistic_regression(X, y, lr=0.1, epochs=500):
        n, d = X.shape
        n_classes = 10
        y_onehot = np.zeros((n, n_classes))
        y_onehot[np.arange(n), y.astype(int)] = 1
        X_bias = np.hstack([np.ones((n, 1)), X])
        W = np.zeros((d + 1, n_classes))
        for epoch in range(epochs):
            scores = X_bias @ W
            max_scores = np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            grad = X_bias.T @ (probs - y_onehot) / n
            W -= lr * grad
        return W
    
    def predict_logistic(W, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        scores = X_bias @ W
        return np.argmax(scores, axis=1)
    
    best_val_acc = 0
    best_model = None
    
    for k in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        cluster_labels, centroids = kmeans(X_all_norm, k, seed=42)
        
        labeled_clusters = cluster_labels[:n_labeled]
        cluster_to_label = {}
        for cid in range(k):
            mask = labeled_clusters == cid
            if np.any(mask):
                labels = y_labeled[mask]
                counts = np.bincount(labels.astype(int), minlength=10)
                cluster_to_label[cid] = np.argmax(counts)
        
        pseudo_labels = np.zeros(X_unlabeled.shape[0], dtype=int)
        for i, cid in enumerate(cluster_labels[n_labeled:]):
            if cid in cluster_to_label:
                pseudo_labels[i] = cluster_to_label[cid]
            else:
                min_dist = float('inf')
                best_label = 0
                for cid2, label in cluster_to_label.items():
                    dist = np.linalg.norm(centroids[cid] - centroids[cid2])
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
                pseudo_labels[i] = best_label
        
        X_train = np.vstack([X_labeled, X_unlabeled])
        y_train = np.concatenate([y_labeled, pseudo_labels])
        
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        train_std[train_std == 0] = 1.0
        X_train_norm = (X_train - train_mean) / train_std
        X_val_norm = (X_val - train_mean) / train_std
        
        W = logistic_regression(X_train_norm, y_train, lr=0.1, epochs=500)
        val_pred = predict_logistic(W, X_val_norm)
        val_acc = np.mean(val_pred == y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = (W, train_mean, train_std)
    
    if best_model is None:
        cluster_labels, centroids = kmeans(X_all_norm, 20, seed=42)
        labeled_clusters = cluster_labels[:n_labeled]
        cluster_to_label = {}
        for cid in range(20):
            mask = labeled_clusters == cid
            if np.any(mask):
                labels = y_labeled[mask]
                counts = np.bincount(labels.astype(int), minlength=10)
                cluster_to_label[cid] = np.argmax(counts)
        
        pseudo_labels = np.zeros(X_unlabeled.shape[0], dtype=int)
        for i, cid in enumerate(cluster_labels[n_labeled:]):
            pseudo_labels[i] = cluster_to_label.get(cid, 0)
        
        X_train = np.vstack([X_labeled, X_unlabeled])
        y_train = np.concatenate([y_labeled, pseudo_labels])
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        train_std[train_std == 0] = 1.0
        X_train_norm = (X_train - train_mean) / train_std
        W = logistic_regression(X_train_norm, y_train, lr=0.1, epochs=500)
        best_model = (W, train_mean, train_std)
    
    W, train_mean, train_std = best_model
    
    def predict(X):
        X_norm = (X - train_mean) / train_std
        return predict_logistic(W, X_norm)
    
    return predict