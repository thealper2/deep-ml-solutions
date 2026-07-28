def residual_summary(y_true, y_pred):
    residuals = y_true - y_pred
    mean = float(np.mean(residuals))
    std = float(np.std(residuals))
    median_abs = float(np.median(np.abs(residuals)))
    return {'mean': mean, 'std': std, 'median_abs': median_abs}
