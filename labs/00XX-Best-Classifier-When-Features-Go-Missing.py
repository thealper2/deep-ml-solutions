import numpy as np
from scipy.special import logsumexp


def train(X_train, y_train, K=5, n_models=8, alpha=0.5, iters=30, seed=0):
    """Ensemble of per-class latent-class mixtures on binned pixels.
    Missing pixels are marginalised out. Temperature is fitted on held-out training rows."""
    edges = [1, 5, 9, 13]
    L, C = 5, 10
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train).astype(int)
    rng = np.random.default_rng(seed)

    def onehot(X):
        b = np.digitize(np.nan_to_num(X), edges)
        O = np.zeros((*X.shape, L))
        i, j = np.where(~np.isnan(X))
        O[i, j, b[i, j]] = 1
        return O

    def fit(O, y, r):
        models = []
        for c in range(C):
            o = O[y == c]
            R = r.dirichlet(np.ones(K), len(o))
            for _ in range(iters):
                t = np.einsum('nk,ndl->kdl', R, o) + alpha
                t /= t.sum(2, keepdims=True)
                w = R.sum(0) + 1
                w /= w.sum()
                l = np.einsum('ndl,kdl->nk', o, np.log(t)) + np.log(w)
                R = np.exp(l - logsumexp(l, axis=1, keepdims=True))
            models.append((np.log(t), np.log(w)))
        return models

    def scores(models, O):
        return np.stack([logsumexp(np.einsum('ndl,kdl->nk', O, t) + w, axis=1) for t, w in models], 1)

    log_prior = np.log((np.bincount(y_train, minlength=C) + 1) / (len(y_train) + C))

    def ens(all_models, O, T):
        out = []
        for m in all_models:
            s = scores(m, O) / T + log_prior
            out.append(s - logsumexp(s, axis=1, keepdims=True))
        return logsumexp(np.stack(out), axis=0) - np.log(len(out))

    idx = rng.permutation(len(X_train))
    cut = int(0.8 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    held = fit(onehot(X_train[tr]), y_train[tr], rng)
    Xv = X_train[va].copy()
    Xv[rng.random(Xv.shape) < rng.random((len(Xv), 1))] = np.nan
    Ov = onehot(Xv)
    grid = [1, 1.25, 1.5, 1.75, 2, 2.5, 3]
    losses = [-ens([held], Ov, T)[np.arange(len(va)), y_train[va]].mean() for T in grid]
    T = grid[int(np.argmin(losses))]

    O = onehot(X_train)
    models = [fit(O, y_train, rng) for _ in range(n_models)]

    def predict_proba(X):
        X = np.asarray(X, float)
        p = np.exp(ens(models, onehot(X), T))
        return p / p.sum(1, keepdims=True)

    return predict_proba
