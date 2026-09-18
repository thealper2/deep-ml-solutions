import numpy as np
from sklearn.model_selection import cross_val_score


def cv_mse(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    mses = -scores
    mean = float(mses.mean())
    se = float(mses.std(ddof=1) / np.sqrt(len(mses)))
    return round(mean, 1), round(se, 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []
    for v in values:
        model = make_model(v)
        mean, se = cv_mse(model, X, y, cv)
        means.append(mean)
        ses.append(se)

    return means, ses


def one_se_rule(values, means, ses, prefer='smaller'):
    values = list(values)
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    candidates = [v for v, m in zip(values, means) if m <= threshold]
    if prefer == 'smaller':
        return min(candidates)

    return max(candidates)