def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    resid_summary = residual_summary(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'residual_summary': resid_summary
    }
