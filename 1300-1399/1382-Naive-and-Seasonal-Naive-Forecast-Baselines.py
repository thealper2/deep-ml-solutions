import numpy as np

def baseline_forecasts(series, season, horizon):
    """Return {"naive": [...], "seasonal_naive": [...]} with horizon floats each."""
    naive = np.array([series[-1]] * horizon).astype(float)
    last_season = series[-season:]
    forecast = (np.tile(last_season, int(np.ceil(horizon / season)))[:horizon]).astype(float)
    return {"naive": naive.tolist(), "seasonal_naive": forecast.tolist()}

def mae(y_true, y_pred):
    """Mean absolute error as a float."""
    return sum(abs(y_t - y_p) for y_t, y_p in zip(y_true, y_pred)) / len(y_true)
