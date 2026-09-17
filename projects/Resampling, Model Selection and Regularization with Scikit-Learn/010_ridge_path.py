import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ridge_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    coefs = []
    for a in alphas:
        model = ridge_model(a)
        model.fit(X, y)
        coefs.append(model.named_steps['ridge'].coef_)
    return np.array(coefs)

def coef_norms(path):
    return np.round(np.linalg.norm(np.asarray(path), axis=1), 2)