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