import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def score_path(X, y, path, cv):
    sizes = []
    means = []
    ses = []
    for k in sorted(path.keys()):
        cols = path[k]
        scores = cross_val_score(
            LinearRegression(),
            X[cols],
            y,
            cv=cv,
            scoring='neg_mean_squared_error',
        )
        mses = -scores
        mean = float(mses.mean())
        se = float(mses.std(ddof=1) / np.sqrt(len(mses)))
        sizes.append(int(k))
        means.append(round(mean, 1))
        ses.append(round(se, 1))
        
    return sizes, means, ses


def best_size(sizes, means):
    idx = int(np.argmin(means))
    return sizes[idx]