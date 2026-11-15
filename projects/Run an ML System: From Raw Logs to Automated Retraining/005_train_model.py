from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X, y, C=1.0):
    """Fitted StandardScaler + LogisticRegression(C, max_iter=2000, tol=1e-8) pipeline."""
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, tol=1e-8)
    )
    model.fit(X, y)
    return model