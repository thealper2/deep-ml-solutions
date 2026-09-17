import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

def cv_mse(estimator, X, y, k=5, random_state=0):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        estimator,
        X, y,
        cv=kf,
        scoring='neg_mean_squared_error'
    )
    mses = -scores
    mean = float(mses.mean())
    se = float(mses.std(ddof=1) / np.sqrt(k))
    return round(mean, 2), round(se, 2)

def loocv_mse(estimator, X, y):
    loo = LeaveOneOut()
    scores = cross_val_score(
        estimator,
        X, y,
        cv=loo,
        scoring='neg_mean_squared_error'
    )
    mses = -scores
    return round(float(mses.mean()), 2)