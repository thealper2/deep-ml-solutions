from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def numeric_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())