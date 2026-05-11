import numpy as np

def k_nearest_neighbors(data, point, k):
    distances = np.linalg.norm(data - point, axis=1)
    nearest_neighbors = np.argsort(distances)[1:k+1]
    return nearest_neighbors, distances[nearest_neighbors]

def local_reachability_distance(data, point_idx, k, distances_to_neighbors):
    neighbors, _ = k_nearest_neighbors(data, data[point_idx], k)
    
    k_distances = []
    for neighbor_idx in neighbors:
        neighbor_distances = np.linalg.norm(data - data[neighbor_idx], axis=1)
        neighbor_distances_sorted = np.sort(neighbor_distances)
        k_distance = neighbor_distances_sorted[k]
        k_distances.append(k_distance)
    
    reachability_distances = np.maximum(distances_to_neighbors, k_distances)
    
    return 1 / (np.mean(reachability_distances))

def local_outlier_factor(X, k):
    """
    Compute the Local Outlier Factor (LOF) score for each point in X.

    Args:
        X: array-like of shape (n, d)
        k: int, neighborhood size

    Returns:
        list of length n with the LOF score of each point
    """
    X = np.array(X)
    n = X.shape[0]
    scores = []

    neighbors_list = []
    distances_list = []
    for i in range(n):
        neighbors, distances = k_nearest_neighbors(X, X[i], k)
        neighbors_list.append(neighbors)
        distances_list.append(distances)

    lrd_values = []
    for i in range(n):
        lrd = local_reachability_distance(X, i, k, distances_list[i])
        lrd_values.append(lrd)

    for i in range(n):
        neighbors = neighbors_list[i]
        lrd_neighbors = [lrd_values[neighbor] for neighbor in neighbors]
        lof = np.mean(lrd_neighbors) / lrd_values[i]
        scores.append(round(lof, 4))

    return scores
