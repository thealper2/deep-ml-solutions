def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    diff = np.abs(analytic_grad - numeric_grad)
    denominator = np.maximum(np.abs(analytic_grad), np.abs(numeric_grad))
    denominator = np.maximum(denominator, tol)
    relative_errors = diff / denominator
    return float(np.max(relative_errors))