import numpy as np

def pairwise_cosine_similarity(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms
    similarity_matrix = np.dot(X_normalized, X_normalized.T)
    similarity_matrix[np.isnan(similarity_matrix)] = 0.0
    similarity_matrix = np.round(similarity_matrix, 4)
    return similarity_matrix.tolist()