def ols_fit(X, y):
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return theta
