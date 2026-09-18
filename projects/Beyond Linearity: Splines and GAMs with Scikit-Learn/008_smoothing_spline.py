import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge


def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots, degree=3, knots='quantile', include_bias=False
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps['splinetransformer']
    ridge = model.named_steps['ridge']
    alpha = float(ridge.alpha)

    B = transformer.transform(X)
    B = np.asarray(B, dtype=float)

    B = B - B.mean(axis=0, keepdims=True)

    p = B.shape[1]
    BtB = B.T @ B
    M = np.linalg.inv(BtB + alpha * np.eye(p)) @ BtB
    df = 1.0 + float(np.trace(M))
    return round(df, 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    alphas = list(alphas)
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer='larger')
    return alpha_min, alpha_1se