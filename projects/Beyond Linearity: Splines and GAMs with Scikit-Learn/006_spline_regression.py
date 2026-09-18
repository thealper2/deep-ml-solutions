import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression


def spline_model(n_knots, degree=3, extrapolation='constant'):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots='quantile',
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps['splinetransformer']
    sample = X.iloc[:3] if hasattr(X, 'iloc') else X[:3]
    out = transformer.transform(sample)
    return int(out.shape[1])


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    knot_counts = list(knot_counts)
    means, ses = spline_curve(X, y, knot_counts, cv)
    k_min = knot_counts[int(np.argmin(means))]
    k_1se = one_se_rule(knot_counts, means, ses, prefer='smaller')
    return k_min, k_1se