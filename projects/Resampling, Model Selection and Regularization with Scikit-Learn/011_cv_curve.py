import numpy as np
from sklearn.model_selection import cross_val_score


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []
    for v in values:
        model = make_model(v)
        scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring='neg_mean_squared_error'
        )
        mses = -scores
        mean = float(mses.mean())
        se = float(mses.std(ddof=1) / np.sqrt(len(mses)))
        means.append(round(mean, 1))
        ses.append(round(se, 1))
        
    return means, ses


def choose_penalty(make_model, X, y, values, cv):
    values = list(values)
    means, ses = cv_curve(make_model, X, y, values, cv)
    i_min = int(np.argmin(means))
    value_min = values[i_min]
    value_1se = one_se_rule(values, means, ses, prefer='larger')
    return value_min, value_1se