import numpy as np
from sklearn.cross_decomposition import PLSRegression


def pls_model(n_components):
    return PLSRegression(n_components=n_components, scale=True)


def pls_curve(X, y, cv):
    p = X.shape[1]
    values = list(range(1, p + 1))
    means, ses = cv_curve(pls_model, X, y, values, cv)
    return values, means, ses


def pls_predict(model, X):
    return np.asarray(model.predict(X)).ravel()


def best_components(components, means, ses):
    comps = list(components)
    m_min = comps[int(np.argmin(means))]
    m_1se = one_se_rule(comps, means, ses, prefer='smaller')
    return m_min, m_1se