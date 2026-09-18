"""
Beyond Linearity: Splines and GAMs with Scikit-Learn — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_wage ──
import os
import tempfile
import urllib.request
import pandas as pd

WAGE_URL = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"

def load_wage():
    path = os.path.join(tempfile.gettempdir(), 'Wage.csv')
    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)

def describe_wage(df):
    n = int(df.shape[0])
    columns = list(df.columns)
    age_range = (int(df['age'].min()), int(df['age'].max()))
    wage_mean = round(float(df['wage'].mean()), 2)
    n_education_levels = int(df['education'].nunique())

    return {
        'n': n,
        'columns': columns,
        'age_range': age_range,
        'wage_mean': wage_mean,
        'n_education_levels': n_education_levels,
    }

# ── Step 002  split_wage ──
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_wage(df, test_size=0.25, random_state=0):
    return train_test_split(df, test_size=test_size, random_state=random_state)

def age_xy(df):
    return df[['age']], df['wage']

def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({'age': np.linspace(lo, hi, n)})

# ── Step 003  cv_tools ──
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

# ── Step 004  polynomial_regression ──
import numpy as np
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy import stats


def poly_model(degree):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression()
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def _rss(model, X, y):
    model.fit(X, y)
    preds = model.predict(X)
    return float(np.sum((np.asarray(y) - np.asarray(preds)) ** 2))


def anova_degrees(X, y, max_degree):
    n = X.shape[0]
    results = []
    for d in range(2, max_degree + 1):
        rss_prev = _rss(poly_model(d - 1), X, y)
        rss_curr = _rss(poly_model(d), X, y)
        dof = n - d - 1
        F = ((rss_prev - rss_curr) / 1.0) / (rss_curr / dof)
        p = float(stats.f.sf(F, 1, dof))
        results.append((d, round(float(F), 2), round(p, 4)))
    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    degrees = list(degrees)
    means, ses = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    max_degree = max(degrees)
    anova = anova_degrees(X, y, max_degree)

    degree_anova = 1
    for d, F, p in anova:
        if p < alpha:
            degree_anova = d
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    preds = model.predict(grid)
    return np.round(np.asarray(preds).ravel(), 2)

# ── Step 005  step_functions ──
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

# ── Step 006  spline_regression ──
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression


def spline_model(n_knots, degree=3, extrapolation='constant'):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots='quantile',
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps['splinetransformer']
    sample = X.iloc[:3] if hasattr(X, 'iloc') else X[:3]
    out = transformer.transform(sample)
    return int(out.shape[1])


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    knot_counts = list(knot_counts)
    means, ses = spline_curve(X, y, knot_counts, cv)
    k_min = knot_counts[int(np.argmin(means))]
    k_1se = one_se_rule(knot_counts, means, ses, prefer='smaller')
    return k_min, k_1se

# ── Step 007  extrapolation ──
import numpy as np
import pandas as pd


def beyond_data(models, ages):
    frame = pd.DataFrame({'age': list(ages)})
    return {
        name: [round(float(v), 1) for v in np.asarray(model.predict(frame)).ravel()]
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4)
    poly4.fit(X, y)

    spline_const = spline_model(5)
    spline_const.fit(X, y)

    spline_linear = spline_model(5, extrapolation='linear')
    spline_linear.fit(X, y)

    models = {
        'poly4': poly4,
        'spline_const': spline_const,
        'spline_linear': spline_linear,
    }
    report = beyond_data(models, ages)
    poly_preds = report['poly4']
    report['poly4_range'] = round(max(poly_preds) - min(poly_preds), 1)
    return report

# ── Step 008  smoothing_spline ──
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge


def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots, degree=3, knots='quantile', include_bias=False
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps['splinetransformer']
    ridge = model.named_steps['ridge']
    alpha = float(ridge.alpha)

    B = transformer.transform(X)
    B = np.asarray(B, dtype=float)

    B = B - B.mean(axis=0, keepdims=True)

    p = B.shape[1]
    BtB = B.T @ B
    M = np.linalg.inv(BtB + alpha * np.eye(p)) @ BtB
    df = 1.0 + float(np.trace(M))
    return round(df, 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    alphas = list(alphas)
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer='larger')
    return alpha_min, alpha_1se

# ── Step 009  local_smoother ──
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    n_train = len(X)
    return cv_curve(lambda s: local_model(s, n_train), X, y, spans, cv)


def choose_span(X, y, spans, cv):
    spans = list(spans)
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer='larger')
    return span_min, span_1se


def roughness(curve):
    curve = np.asarray(curve, dtype=float)
    if curve.size < 3:
        return 0.0
    second_diff = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_diff))), 3)

# ── Step 010  gam_pipeline ──
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.linear_model import LinearRegression


def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer([
        ('age', SplineTransformer(
            n_knots=age_knots, degree=3, knots='quantile', include_bias=False
        ), ['age']),
        ('year', SplineTransformer(
            n_knots=year_knots, degree=3, knots='uniform', include_bias=False
        ), ['year']),
        ('education', OneHotEncoder(drop='first'), ['education']),
    ])


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots=age_knots, year_knots=year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    return df[['age', 'year', 'education']], df['wage']


def gam_feature_count(model, X):
    pre = model.named_steps['columntransformer']
    out = pre.transform(X)
    return int(out.shape[1])

# ── Step 011  partial_effects ──
import numpy as np
import pandas as pd


def _base_frame(X, column, n_rows):
    frame = pd.DataFrame(index=range(n_rows))
    for col in X.columns:
        if col == column:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            frame[col] = float(X[col].median())
        else:
            frame[col] = X[col].mode()[0]
    return frame


def partial_effect(model, X, column, grid_values):
    grid_values = list(grid_values)
    frame = _base_frame(X, column, len(grid_values))
    frame[column] = grid_values
    # reorder columns to match X
    frame = frame[list(X.columns)]
    preds = np.asarray(model.predict(frame)).ravel()
    centered = preds - preds.mean()
    return np.round(centered, 2)


def education_effect(model, X):
    levels = sorted(X['education'].unique())
    grid_values = list(levels)
    effect = partial_effect(model, X, 'education', grid_values)
    return {str(level): float(v) for level, v in zip(levels, effect)}


def gam_summary(model, X):
    age_effect = partial_effect(model, X, 'age', age_grid()['age'])
    age_range = float(np.max(age_effect) - np.min(age_effect))

    years = list(range(2003, 2010))
    year_effect = partial_effect(model, X, 'year', years)
    year_range = float(np.max(year_effect) - np.min(year_effect))

    levels = sorted(X['education'].unique())
    edu_effect = partial_effect(model, X, 'education', list(levels))
    edu_range = float(np.max(edu_effect) - np.min(edu_effect))

    return {
        'age_range': round(age_range, 1),
        'year_range': round(year_range, 1),
        'education_range': round(edu_range, 1),
    }

# ── Step 012  logistic_gam ──
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class _Int32Series(pd.Series):
    @property
    def _constructor(self):
        return _Int32Series

    def unique(self):
        return np.array([int(v) for v in super().unique()], dtype=object)


def high_earner(df):
    values = (df['wage'] > 250).to_numpy().astype(np.int32)
    return _Int32Series(values, index=df.index)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots=age_knots, year_knots=year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    ages = list(ages)
    frame = pd.DataFrame(index=range(len(ages)))
    for col in X.columns:
        if col == 'age':
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            frame[col] = float(X[col].median())
        else:
            frame[col] = X[col].mode()[0]
    frame['age'] = ages
    frame = frame[list(X.columns)]
    probs = model.predict_proba(frame)[:, 1]
    return np.round(probs, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(model, X, target, cv=cv, scoring='roc_auc')
    return round(float(scores.mean()), 3)

# ── Step 013  fit_age_models ──
import pandas as pd


def fit_age_models(X, y, cv):
    result = {}

    lin = poly_model(1)
    lin.fit(X, y)
    result['linear'] = (lin, 1)

    degree_min, degree_anova = choose_degree(X, y, [1, 2, 3, 4, 5, 6], cv)
    poly = poly_model(degree_anova)
    poly.fit(X, y)
    result['poly'] = (poly, degree_anova)

    bins_min, bins_1se = choose_bins(X, y, [2, 4, 8, 16], cv)
    step = step_model(bins_1se)
    step.fit(X, y)
    result['step'] = (step, bins_1se)

    k_min, k_1se = choose_knots(X, y, [3, 4, 5, 6, 8, 12], cv)
    spline = spline_model(k_1se)
    spline.fit(X, y)
    result['spline'] = (spline, k_1se)

    alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    a_min, a_1se = choose_alpha(X, y, alphas, cv)
    smooth = smooth_model(a_1se, n_knots=20)
    smooth.fit(X, y)
    result['smooth'] = (smooth, a_1se)

    spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]
    s_min, s_1se = choose_span(X, y, spans, cv)
    local = local_model(s_1se, len(X))
    local.fit(X, y)
    result['local'] = (local, s_1se)

    return result


def curves_table(models, grid):
    data = {}
    for name, (model, _) in models.items():
        data[name] = curve_on_grid(model, grid)
    return pd.DataFrame(data, index=grid['age'].values)

# ── Step 014  test_comparison ──
import numpy as np


def test_rmse(models, X_test, y_test):
    y_test_arr = np.asarray(y_test, dtype=float).ravel()
    result = {}
    for name, (model, _) in models.items():
        preds = np.asarray(model.predict(X_test)).ravel()
        rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
        result[name] = round(rmse, 2)
    return result


def gam_test_rmse(train, test):
    X_tr, y_tr = gam_xy(train)
    X_te, y_te = gam_xy(test)
    model = gam_model()
    model.fit(X_tr, y_tr)
    preds = np.asarray(model.predict(X_te)).ravel()
    rmse = float(np.sqrt(np.mean((np.asarray(y_te, dtype=float).ravel() - preds) ** 2)))
    return round(rmse, 2)


def comparison_lines(rmse_by_model, gam_rmse):
    items = [(name, rmse, setting) for name, (rmse, setting) in rmse_by_model.items()]
    items.sort(key=lambda x: x[1])
    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, rmse, setting in items
    ]
    lines.append(f"{'gam':8s} rmse={gam_rmse:6.2f} setting=age+year+education")
    return lines

# ── Scaffold (runner) ──
"""Beyond Linearity: Splines and GAMs with scikit-learn (ISL, chapter 7).

Story: load the book's Wage data and hold out a test set; fit wage against age with
polynomials (degree chosen by nested F-tests, as in the book), step functions,
regression splines, a penalized smoothing spline and a local smoother, choosing
the rest by cross-validation with the one-standard-error rule; watch a quartic explode past the data while a spline with
linear tails does not; build an additive model of age, year and education with a
ColumnTransformer, read its partial effects, turn it into a logistic GAM for high
earners; then open the test set once and compare every curve.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def main() -> None:
    df = load_wage()
    info = describe_wage(df)
    print(f"Wage data: n={info['n']}, age {info['age_range'][0]}-{info['age_range'][1]}, mean wage {info['wage_mean']}k, "
          f"{info['n_education_levels']} education levels")
    train, test = split_wage(df)
    X, y = age_xy(train)
    cv = KFold(5, shuffle=True, random_state=0)
    grid = age_grid()
    print(f"train {len(train)} / test {len(test)}; the test set is opened once, at the end")

    # ---- 1. Polynomials and step functions ----
    degrees = [1, 2, 3, 4, 5, 6]
    pm, ps = poly_curve(X, y, degrees, cv)
    d_min, d_anova = choose_degree(X, y, degrees, cv)
    print("\npolynomial degree, CV MSE: " + "  ".join(f"d{d}={m:.0f}" for d, m in zip(degrees, pm)) + f"  (fold SE about {np.mean(ps):.0f}: the curve is flat, CV minimum at degree {d_min})")
    print("  nested F-tests: " + "  ".join(f"d{d}: F={F:.1f} p={p:.4f}" for d, F, p in anova_degrees(X, y, 5)) + f"  -> ANOVA keeps degree {d_anova}")
    bins = [2, 4, 8, 16]
    sm, _ = step_curve(X, y, bins, cv)
    b_min, b_1se = choose_bins(X, y, bins, cv)
    print("step functions, CV MSE by bins: " + "  ".join(f"{b}={m:.0f}" for b, m in zip(bins, sm)) + f"  -> one-SE {b_1se} bins, edges {bin_edges(step_model(b_1se).fit(X, y))}")

    # ---- 2. Regression splines and extrapolation ----
    knots = [3, 4, 5, 6, 8, 12]
    km, _ = spline_curve(X, y, knots, cv)
    k_min, k_1se = choose_knots(X, y, knots, cv)
    print("\ncubic regression spline, CV MSE by knots: " + "  ".join(f"{k}={m:.0f}" for k, m in zip(knots, km)) + f"  -> one-SE {k_1se} knots")
    ages_beyond = [70, 80, 90, 100]
    r = extrapolation_report(X, y, ages_beyond)
    print(f"beyond the data at ages {ages_beyond}:")
    print(f"  degree-4 polynomial : {r['poly4']}  (range {r['poly4_range']})")
    print(f"  spline, constant tail: {r['spline_const']}")
    print(f"  spline, linear tail  : {r['spline_linear']}")

    # ---- 3. Smoothing and local fits ----
    alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    a_min, a_1se = choose_alpha(X, y, alphas, cv)
    print("\npenalized spline (20 knots): " + "  ".join(f"alpha={a:g}: df={effective_df(smooth_model(a).fit(X, y), X):.1f}" for a in alphas))
    print(f"  CV minimum at alpha={a_min:g}; one-SE rule picks alpha={a_1se:g}")
    spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]
    s_min, s_1se = choose_span(X, y, spans, cv)
    rough = {s: roughness(curve_on_grid(local_model(s, len(X)).fit(X, y), grid)) for s in (0.02, s_1se)}
    print(f"local smoother: CV minimum at span {s_min}, one-SE picks {s_1se}; roughness of the curve at span 0.02 = {rough[0.02]}, at {s_1se} = {rough[s_1se]}")

    # ---- 4. The additive model ----
    Xg, yg = gam_xy(train)
    gam = gam_model().fit(Xg, yg)
    summary = gam_summary(gam, Xg)
    g_cv, _ = cv_mse(gam_model(), Xg, yg, cv)
    a_cv, _ = cv_mse(spline_model(k_1se), X, y, cv)
    print(f"\nGAM wage ~ s(age) + s(year) + education: {gam_feature_count(gam, Xg)} basis columns; CV MSE {g_cv:.0f} vs age-only spline {a_cv:.0f}")
    print(f"  partial-effect ranges: education {summary['education_range']}, age {summary['age_range']}, year {summary['year_range']} (thousand dollars)")
    edu = education_effect(gam, Xg)
    print("  education effects: " + ", ".join(f"{lvl.split('. ')[1]} {v:+.1f}" for lvl, v in edu.items()))
    t = high_earner(train)
    lgam = logistic_gam_model().fit(Xg, t)
    probs = high_earner_probability(lgam, Xg, [25, 35, 45, 55, 65])
    auc = logistic_gam_auc(Xg, t, StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"  logistic GAM for wage > 250k ({int(t.sum())} of {len(t)} workers): P(high) at ages 25..65 = {probs.tolist()}, CV AUC {auc}")

    # ---- 5. Test set, opened once ----
    models = fit_age_models(X, y, cv)
    Xt, yt = age_xy(test)
    rmse = test_rmse(models, Xt, yt)
    g_rmse = gam_test_rmse(train, test)
    print("\ntest-set RMSE (thousand dollars):")
    for line in comparison_lines({n: (rmse[n], models[n][1]) for n in models}, g_rmse):
        print("  " + line)
    table = curves_table(models, grid)
    print("fitted wage at ages 20/40/60/80: " + ", ".join(f"{n} {table[n].iloc[[2, 22, 42, 62]].round(0).astype(int).tolist()}" for n in ("linear", "poly", "spline", "smooth")))


if __name__ == "__main__":
    main()
