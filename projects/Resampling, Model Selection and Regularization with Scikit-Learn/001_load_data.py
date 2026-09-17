import pandas as pd
from sklearn.datasets import load_diabetes

def load_data():
    data = load_diabetes(as_frame=True)
    return data.data, data.target

def describe_data(X, y):
    n = X.shape[0]
    p = X.shape[1]
    features = list(X.columns)
    y_mean = np.round(float(np.mean(y)), 2)
    return {
        'n': n,
        'p': p,
        'features': features,
        'y_mean': y_mean,
    }