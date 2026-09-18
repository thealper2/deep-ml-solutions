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