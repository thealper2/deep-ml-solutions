from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

def select_features(X, y, k, direction, cv):
    selector = SequentialFeatureSelector(
        LinearRegression(),
        n_features_to_select=k,
        direction=direction,
        cv=cv,
        scoring='neg_mean_squared_error',
    )
    selector.fit(X, y)
    return list(X.columns[selector.get_support()])

def stepwise_path(X, y, direction, cv):
    p = X.shape[1]
    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k, direction, cv)

    path[p] = list(X.columns)
    return path