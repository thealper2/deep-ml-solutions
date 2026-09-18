import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy='uniform'),
        LinearRegression(),
    )

def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)

def choose_bins(X, y, bins, cv):
    bins = list(bins)
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer='smaller')
    return bins_min, bins_1se

def bin_edges(model):
    disc = model.named_steps['kbinsdiscretizer']
    edges = np.asarray(disc.bin_edges_[0])
    return [round(float(e), 1) for e in edges]

def step_levels(model, grid):
    preds = model.predict(grid)
    levels = np.unique(np.round(np.asarray(preds).ravel(), 1))
    return [float(v) for v in levels]
