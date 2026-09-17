import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV


def lasso_model(alpha):
    return make_pipeline(
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=20000),
    )


def lasso_path(X, y, alphas):
    coefs = []
    for a in alphas:
        model = lasso_model(a)
        model.fit(X, y)
        coefs.append(model.named_steps['lasso'].coef_)
    return np.array(coefs)


def nonzero_features(coef, names, tol=1e-8):
    coef = np.asarray(coef)
    return [n for n, c in zip(names, coef) if abs(c) > tol]


def lasso_cv(X, y, cv):
    model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, max_iter=20000, random_state=0),
    )
    model.fit(X, y)
    lasso = model.named_steps['lassocv']
    alpha = round(float(lasso.alpha_), 4)
    selected = nonzero_features(lasso.coef_, list(X.columns))
    return alpha, selected