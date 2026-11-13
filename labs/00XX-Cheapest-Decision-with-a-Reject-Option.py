import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def _is_cat(col):
    return isinstance(col.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(col)


def _encode(X, cols, vocab):
    """Numeric columns as float; categorical columns as integer codes (missing/unseen -> code 0)."""
    out = np.zeros((len(X), len(cols)))
    for j, c in enumerate(cols):
        col = X[c]
        if c in vocab:
            s = col.astype(object).where(col.notna(), None)
            out[:, j] = [vocab[c].get(v, 0) for v in s]
        else:
            out[:, j] = col.astype(float).to_numpy()
    return out


def _hgb(seed, cat_mask):
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=cat_mask, random_state=seed)


def train(X_train, y_train):
    cols = list(X_train.columns)
    vocab = {}
    for c in cols:
        if _is_cat(X_train[c]):
            vals = sorted(str(v) for v in X_train[c].dropna().unique())
            vocab[c] = {v: i + 1 for i, v in enumerate(vals)}
    vocab = {c: {**d, **{k: d[str(k)] for k in X_train[c].dropna().unique() if str(k) in d}} for c, d in vocab.items()}
    cat_mask = np.array([c in vocab for c in cols])
    Z = _encode(X_train, cols, vocab)
    y = np.asarray(y_train).astype(int)

    oof = np.zeros(len(y))
    models = []
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y)):
        m = _hgb(k, cat_mask).fit(Z[a], y[a])
        oof[b] = m.predict_proba(Z[b])[:, 1]
        models.append(m)
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(oof, y)
    return {"models": models, "iso": iso, "cols": cols, "vocab": vocab}


def predict_p1(model, X):
    Z = _encode(X, model["cols"], model["vocab"])
    p = np.mean([m.predict_proba(Z)[:, 1] for m in model["models"]], axis=0)
    return model["iso"].predict(p)


def decide(model, X, costs):
    p1 = predict_p1(model, X)
    fp, fn, rj = float(costs["c_fp"]), float(costs["c_fn"]), float(costs["c_reject"])
    risk = np.stack([fn * p1, fp * (1 - p1), np.full_like(p1, rj)], axis=1)
    return np.argmin(risk, axis=1).astype(int)