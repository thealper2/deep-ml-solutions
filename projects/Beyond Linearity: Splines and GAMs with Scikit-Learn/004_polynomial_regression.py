import numpy as np
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy import stats

def poly_model(degree):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def _rss(model, X, y):
    model.fit(X, y)
    preds = model.predict(X)
    return float(np.sum((np.asarray(y) - np.asarray(preds)) ** 2))


def anova_degrees(X, y, max_degree):
    n = X.shape[0]
    results = []
    for d in range(2, max_degree + 1):
        rss_prev = _rss(poly_model(d - 1), X, y)
        rss_curr = _rss(poly_model(d), X, y)
        dof = n - d - 1
        F = ((rss_prev - rss_curr) / 1.0) / (rss_curr / dof)
        p = float(stats.f.sf(F, 1, dof))
        results.append((d, round(float(F), 2), round(p, 4)))
    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    degrees = list(degrees)
    means, ses = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    max_degree = max(degrees)
    anova = anova_degrees(X, y, max_degree)

    degree_anova = 1
    for d, F, p in anova:
        if p < alpha:
            degree_anova = d
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    preds = model.predict(grid)
    return np.round(np.asarray(preds).ravel(), 2)
