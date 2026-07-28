def r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
        
    r_s = 1 - (ss_res / ss_tot)
    return r_s if not np.isnan(r_s) else 0.0
