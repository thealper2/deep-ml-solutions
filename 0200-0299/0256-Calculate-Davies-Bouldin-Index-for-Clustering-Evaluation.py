import numpy as np

def davies_bouldin_index(X, labels):
    """
    Calculate the Davies-Bouldin Index for clustering evaluation.
    
    Parameters:
    X: numpy array of shape (n_samples, n_features) - data points
    labels: numpy array of shape (n_samples,) - cluster labels
    
    Returns:
    float: Davies-Bouldin Index rounded to 4 decimal places
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    if num_clusters <= 1:
        return 0.0

    centroids = {}
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroids[label] = np.mean(cluster_points, axis=0)

    scatter = {}
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = centroids[label]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        scatter[label] = np.mean(distances)

    db_sum = 0
    for i in range(num_clusters):
        max_ratio = -np.inf
        for j in range(num_clusters):
            if i == j:
                continue
            
            centroid_dist = np.linalg.norm(centroids[unique_labels[i]] - centroids[unique_labels[j]])
            ratio = (scatter[unique_labels[i]] + scatter[unique_labels[j]]) / centroid_dist
            max_ratio = max(max_ratio, ratio)

        db_sum += max_ratio

    dbi = db_sum / num_clusters
    return round(dbi, 4)
