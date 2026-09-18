import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    n_train = len(X)
    return cv_curve(lambda s: local_model(s, n_train), X, y, spans, cv)


def choose_span(X, y, spans, cv):
    spans = list(spans)
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer='larger')
    return span_min, span_1se


def roughness(curve):
    curve = np.asarray(curve, dtype=float)
    if curve.size < 3:
        return 0.0
    second_diff = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_diff))), 3)