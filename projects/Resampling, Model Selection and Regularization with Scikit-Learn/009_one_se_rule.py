import numpy as np

def one_se_rule(values, means, ses, prefer='smaller'):
    i_min = int(np.argmin(means))
    threshold = means[i_min] + ses[i_min]
    candidates = [v for v, m in zip(values, means) if m <= threshold]
    return min(candidates) if prefer == 'smaller' else max(candidates)

def choose_subset(X, y, direction, cv):
    path = stepwise_path(X, y, direction, cv)
    sizes, means, ses = score_path(X, y, path, cv)
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer='smaller')
    features_1se = path[size_1se]
    return size_min, size_1se, features_1se