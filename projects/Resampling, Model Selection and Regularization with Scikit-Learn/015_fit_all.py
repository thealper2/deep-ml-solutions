import numpy as np
from sklearn.linear_model import LinearRegression


def fit_all(X, y, cv, alphas):
    result = {}

    ols = LinearRegression()
    ols.fit(X, y)
    result['ols'] = (ols, list(X.columns), None)

    size_min, size_1se, features_1se = choose_subset(X, y, 'forward', cv)
    fwd = LinearRegression()
    fwd.fit(X[features_1se], y)
    result['forward_1se'] = (fwd, features_1se, size_1se)

    alpha_min, alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    ridge = ridge_model(alpha_1se)
    ridge.fit(X, y)
    result['ridge_1se'] = (ridge, list(X.columns), alpha_1se)

    alpha_lasso, selected_lasso = lasso_cv(X, y, cv)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    lasso_pipe = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, max_iter=20000, random_state=0),
    )
    lasso_pipe.fit(X, y)
    result['lasso_cv'] = (lasso_pipe, list(X.columns), alpha_lasso)

    comps_pcr, means_pcr, ses_pcr = pcr_curve(X, y, cv)
    _, m_pcr = best_components(comps_pcr, means_pcr, ses_pcr)
    pcr = pcr_model(m_pcr)
    pcr.fit(X, y)
    result['pcr_1se'] = (pcr, list(X.columns), m_pcr)

    comps_pls, means_pls, ses_pls = pls_curve(X, y, cv)
    _, m_pls = best_components(comps_pls, means_pls, ses_pls)
    pls = pls_model(m_pls)
    pls.fit(X, y)
    result['pls_1se'] = (pls, list(X.columns), m_pls)

    return result