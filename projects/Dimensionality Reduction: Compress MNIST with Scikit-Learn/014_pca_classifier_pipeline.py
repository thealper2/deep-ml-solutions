from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(n_components=variance, random_state=random_state),
        LogisticRegression(max_iter=2000),
    )