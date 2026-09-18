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