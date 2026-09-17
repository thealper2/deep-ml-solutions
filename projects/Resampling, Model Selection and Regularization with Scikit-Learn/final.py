"""
Resampling, Model Selection and Regularization with Scikit-Learn — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_data ──
import pandas as pd
from sklearn.datasets import load_diabetes

def load_data():
    data = load_diabetes(as_frame=True)
    return data.data, data.target

def describe_data(X, y):
    n = X.shape[0]
    p = X.shape[1]
    features = list(X.columns)
    y_mean = np.round(float(np.mean(y)), 2)
    return {
        'n': n,
        'p': p,
        'features': features,
        'y_mean': y_mean,
    }

# ── Step 002  train_test ──
from sklearn.model_selection import train_test_split

def train_test(X, y, test_size=0.25, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# ── Step 003  validation_set_curve ──
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def poly_model(degree):
    poly = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression(),
    )
    return poly

def validation_set_curve(X, y, feature, degrees, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X[[feature]], y, test_size=0.5, random_state=random_state)
    curve = {}
    for degree in degrees:
        model = poly_model(degree)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        curve[degree] = round(float(mse), 1)

    return curve

def curve_spread(X, y, feature, degrees, seeds):
    spread = {}
    for d in degrees:
        vals = []
        for s in seeds:
            curve = validation_set_curve(X, y, feature, [d], random_state=s)
            vals.append(curve[d])
        spread[d] = round(max(vals) - min(vals), 1)

    return spread

# ── Step 004  cv_mse ──
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

def cv_mse(estimator, X, y, k=5, random_state=0):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        estimator,
        X, y,
        cv=kf,
        scoring='neg_mean_squared_error'
    )
    mses = -scores
    mean = float(mses.mean())
    se = float(mses.std(ddof=1) / np.sqrt(k))
    return round(mean, 2), round(se, 2)

def loocv_mse(estimator, X, y):
    loo = LeaveOneOut()
    scores = cross_val_score(
        estimator,
        X, y,
        cv=loo,
        scoring='neg_mean_squared_error'
    )
    mses = -scores
    return round(float(mses.mean()), 2)

# ── Step 005  cv_spread_by_k ──
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

# ── Step 006  bootstrap_coefficients ──
import numpy as np
from sklearn.linear_model import LinearRegression


def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = X_arr.shape[0]
    rng = np.random.default_rng(random_state)
    p = X_arr.shape[1]
    coefs = np.empty((n_boot, p), dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        model = LinearRegression()
        model.fit(X_arr[idx], y_arr[idx])
        coefs[i] = model.coef_
    return coefs


def bootstrap_se(coefs):
    return np.round(np.std(np.asarray(coefs), axis=0, ddof=1), 2)


def ols_standard_errors(X, y):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = X_arr.shape[0]
    p = X_arr.shape[1]
    A = np.hstack([np.ones((n, 1)), X_arr])
    coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    resid = y_arr - A @ coef
    rss = float(np.sum(resid ** 2))
    sigma2 = rss / (n - p - 1)
    ATA_inv = np.linalg.inv(A.T @ A)
    se = np.sqrt(sigma2 * np.diag(ATA_inv))
    return np.round(se[1:], 2)

# ── Step 007  stepwise_path ──
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

# ── Step 008  score_path ──
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

# ── Step 009  one_se_rule ──
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

# ── Step 010  ridge_path ──
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ridge_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))

def ridge_path(X, y, alphas):
    coefs = []
    for a in alphas:
        model = ridge_model(a)
        model.fit(X, y)
        coefs.append(model.named_steps['ridge'].coef_)
    return np.array(coefs)

def coef_norms(path):
    return np.round(np.linalg.norm(np.asarray(path), axis=1), 2)

# ── Step 011  cv_curve ──
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

# ── Step 012  lasso_path ──
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV


def lasso_model(alpha):
    return make_pipeline(
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=20000),
    )


def lasso_path(X, y, alphas):
    coefs = []
    for a in alphas:
        model = lasso_model(a)
        model.fit(X, y)
        coefs.append(model.named_steps['lasso'].coef_)
    return np.array(coefs)


def nonzero_features(coef, names, tol=1e-8):
    coef = np.asarray(coef)
    return [n for n, c in zip(names, coef) if abs(c) > tol]


def lasso_cv(X, y, cv):
    model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, max_iter=20000, random_state=0),
    )
    model.fit(X, y)
    lasso = model.named_steps['lassocv']
    alpha = round(float(lasso.alpha_), 4)
    selected = nonzero_features(lasso.coef_, list(X.columns))
    return alpha, selected

# ── Step 013  pcr_model ──
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def pcr_model(n_components):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        LinearRegression(),
    )


def explained_variance(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(Xs)
    return np.round(np.cumsum(pca.explained_variance_ratio_), 3)


def pcr_curve(X, y, cv):
    p = X.shape[1]
    values = list(range(1, p + 1))
    means, ses = cv_curve(pcr_model, X, y, values, cv)
    return values, means, ses

# ── Step 014  pls_model ──
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def pls_model(n_components):
    return PLSRegression(n_components=n_components, scale=True)


def pls_curve(X, y, cv):
    p = X.shape[1]
    values = list(range(1, p + 1))
    means, ses = cv_curve(pls_model, X, y, values, cv)
    return values, means, ses


def pls_predict(model, X):
    return np.asarray(model.predict(X)).ravel()


def best_components(components, means, ses):
    comps = list(components)
    m_min = comps[int(np.argmin(means))]
    m_1se = one_se_rule(comps, means, ses, prefer='smaller')
    return m_min, m_1se

# ── Step 015  fit_all ──
import numpy as np
from sklearn.linear_model import LinearRegression


def fit_all(X, y, cv, alphas):
    result = {}

    ols = LinearRegression()
    ols.fit(X, y)
    result['ols'] = (ols, list(X.columns), None)

    size_min, size_1se, features_1se = choose_subset(X, y, 'forward', cv)
    fwd = LinearRegression()
    fwd.fit(X[features_1se], y)
    result['forward_1se'] = (fwd, features_1se, size_1se)

    alpha_min, alpha_1se = choose_penalty(ridge_model, X, y, alphas, cv)
    ridge = ridge_model(alpha_1se)
    ridge.fit(X, y)
    result['ridge_1se'] = (ridge, list(X.columns), alpha_1se)

    alpha_lasso, selected_lasso = lasso_cv(X, y, cv)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV
    lasso_pipe = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, max_iter=20000, random_state=0),
    )
    lasso_pipe.fit(X, y)
    result['lasso_cv'] = (lasso_pipe, list(X.columns), alpha_lasso)

    comps_pcr, means_pcr, ses_pcr = pcr_curve(X, y, cv)
    _, m_pcr = best_components(comps_pcr, means_pcr, ses_pcr)
    pcr = pcr_model(m_pcr)
    pcr.fit(X, y)
    result['pcr_1se'] = (pcr, list(X.columns), m_pcr)

    comps_pls, means_pls, ses_pls = pls_curve(X, y, cv)
    _, m_pls = best_components(comps_pls, means_pls, ses_pls)
    pls = pls_model(m_pls)
    pls.fit(X, y)
    result['pls_1se'] = (pls, list(X.columns), m_pls)

    return result

# ── Step 016  test_report ──
import numpy as np


def test_report(models, X_test, y_test):
    y_test_arr = np.asarray(y_test, dtype=float).ravel()
    report = {}
    for name, (model, features, setting) in models.items():
        preds = np.asarray(model.predict(X_test[features])).ravel()
        rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
        rmse = round(rmse, 1)

        if name == 'lasso_cv':
            lasso = model.named_steps['lassocv']
            n_features = int(np.sum(np.abs(lasso.coef_) > 1e-8))
        elif name == 'forward_1se':
            n_features = len(features)
        else:
            n_features = len(features)

        report[name] = {
            'rmse': rmse,
            'n_features': n_features,
            'setting': setting,
        }
    return report


def best_method(report):
    return min(report, key=lambda name: report[name]['rmse'])


def format_table(report):
    lines = []
    for name in sorted(report, key=lambda n: report[n]['rmse']):
        entry = report[name]
        lines.append(
            f"{name:12s} rmse={entry['rmse']:6.1f} "
            f"features={entry['n_features']:2d} setting={entry['setting']}"
        )
    return lines

# ── Scaffold (runner) ──
"""Resampling, Model Selection and Regularization with scikit-learn (ISL, chapters 5 and 6).

Story: load the diabetes data and hold out a test set; watch the validation-set
estimate move with the split, then replace it with k-fold and leave-one-out
cross-validation; get coefficient standard errors from the bootstrap and check them
against the formula; run forward stepwise selection and pick a size with the
one-standard-error rule; trace ridge and lasso paths and pick penalties the same
way; build PCR and PLS pipelines and pick component counts; finally open the test
set once and compare every method in one table.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def main() -> None:
    X, y = load_data()
    info = describe_data(X, y)
    print(f"diabetes data: n={info['n']} p={info['p']} features={info['features']} mean progression {info['y_mean']}")
    X_tr, X_te, y_tr, y_te = train_test(X, y)
    cv = KFold(5, shuffle=True, random_state=0)
    print(f"train {len(X_tr)} / test {len(X_te)} (the test set is opened once, at the end)")

    # ---- 1. One split lies ----
    degrees = [1, 2, 3, 4]
    curves = [validation_set_curve(X_tr, y_tr, 'bmi', degrees, s) for s in range(3)]
    print("\nvalidation-set MSE of poly(bmi) by degree, three different 50/50 splits:")
    for s, c in enumerate(curves):
        print(f"  seed {s}: " + "  ".join(f"d{d}={c[d]:.0f}" for d in degrees) + f"  -> best degree {min(c, key=c.get)}")
    spread = curve_spread(X_tr, y_tr, 'bmi', degrees, seeds=range(8))
    print("  range across 8 splits: " + "  ".join(f"d{d}={spread[d]:.0f}" for d in degrees))

    # ---- 2. Cross-validation and the bootstrap ----
    r = compare_with_loocv(LinearRegression(), X_tr, y_tr, ks=[2, 5, 10], seeds=range(6))
    print("\nfull linear model, CV estimate of MSE (mean over 6 fold assignments, sd across them):")
    for k, (m, sd) in r['kfold'].items():
        print(f"  {k:2d}-fold: {m:.0f} (sd {sd:.0f})")
    print(f"  LOOCV : {r['loocv']:.0f} (no randomness)")
    coefs = bootstrap_coefficients(X_tr, y_tr, n_boot=300)
    se_b, se_f = bootstrap_se(coefs), ols_standard_errors(X_tr, y_tr)
    print("bootstrap vs formula standard errors: " + ", ".join(f"{n} {b:.0f}/{f:.0f}" for n, b, f in zip(X.columns, se_b, se_f)))

    # ---- 3. Subset selection ----
    size_min, size_1se, feats = choose_subset(X_tr, y_tr, 'forward', cv)
    print(f"\nforward stepwise: CV minimum at {size_min} features; one-SE rule keeps {size_1se}: {feats}")

    # ---- 4. Ridge and the lasso ----
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    means, ses = cv_curve(ridge_model, X_tr, y_tr, alphas, cv)
    a_min, a_1se = choose_penalty(ridge_model, X_tr, y_tr, alphas, cv)
    print("\nridge CV curve: " + "  ".join(f"a={a:g}:{m:.0f}" for a, m in zip(alphas, means)))
    print(f"  minimum at alpha={a_min:g}, one-SE rule picks alpha={a_1se:g}; coefficient norms along the path: {coef_norms(ridge_path(X_tr, y_tr, alphas)).tolist()}")
    lasso_alphas = [0.01, 0.5, 2.0, 5.0, 20.0]
    counts = [len(nonzero_features(row, list(X.columns))) for row in lasso_path(X_tr, y_tr, lasso_alphas)]
    alpha_l, selected = lasso_cv(X_tr, y_tr, cv)
    print("lasso nonzero coefficients along the path: " + "  ".join(f"a={a:g}:{c}" for a, c in zip(lasso_alphas, counts)))
    print(f"  LassoCV alpha={alpha_l}: keeps {len(selected)} features {selected}")

    # ---- 5. PCR and PLS ----
    ev = explained_variance(X_tr)
    comps, pm, ps = pcr_curve(X_tr, y_tr, cv)
    _, lm, ls = pls_curve(X_tr, y_tr, cv)
    print("\ncomponents:      " + " ".join(f"{c:5d}" for c in comps))
    print("cum. variance:   " + " ".join(f"{v:5.2f}" for v in ev))
    print("PCR CV MSE:      " + " ".join(f"{m:5.0f}" for m in pm))
    print("PLS CV MSE:      " + " ".join(f"{m:5.0f}" for m in lm))
    print(f"one-SE component counts: PCR {best_components(comps, pm, ps)[1]}, PLS {best_components(comps, lm, ls)[1]}")

    # ---- 6. One honest table ----
    models = fit_all(X_tr, y_tr, cv, alphas)
    report = test_report(models, X_te, y_te)
    print("\ntest set, opened once:")
    for line in format_table(report):
        print("  " + line)
    print(f"best on the test set: {best_method(report)}; the regularized models sit within a few units of each other, "
          f"and the sparsest one pays a little accuracy for using only {report['forward_1se']['n_features']} features")


if __name__ == "__main__":
    main()
