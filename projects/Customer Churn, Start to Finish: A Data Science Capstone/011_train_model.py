from sklearn.linear_model import LogisticRegression


def train_model(X, y, C=1.0):
    """Fit and return LogisticRegression(C=C, max_iter=2000, tol=1e-8)."""
    model = LogisticRegression(C=C, max_iter=2000, tol=1e-8)
    model.fit(X, y)
    return model