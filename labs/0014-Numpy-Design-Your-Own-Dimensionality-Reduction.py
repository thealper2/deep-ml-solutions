import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.
    
    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        self.components = None
        self.mean = None
    
    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        X = np.array(X)
        n_samples = X.shape[0]

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, :self.n_components]

        return self

    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        X = np.array(X)
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
