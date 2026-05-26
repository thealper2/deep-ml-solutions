import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    X = np.array(X)
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    total_variance = np.sum(eigenvalues_sorted)
    ratios = eigenvalues_sorted / total_variance
    ratios = np.round(ratios, 10)
    return ratios.tolist()