from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

def multiclass_pipeline(random_state=42):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )