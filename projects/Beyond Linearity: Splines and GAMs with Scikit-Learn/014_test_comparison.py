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