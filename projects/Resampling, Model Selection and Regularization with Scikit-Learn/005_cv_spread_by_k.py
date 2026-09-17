import numpy as np

def cv_spread_by_k(estimator, X, y, ks, seeds):
    result = {}
    for k in ks:
        vals = [cv_mse(estimator, X, y, k=k, random_state=s)[0] for s in seeds]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))
        result[k] = (round(mean, 1), round(std, 1))

    return result

def compare_with_loocv(estimator, X, y, ks, seeds):
    return {
        'loocv': loocv_mse(estimator, X, y),
        'kfold': cv_spread_by_k(estimator, X, y, ks, seeds),
    }