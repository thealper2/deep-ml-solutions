import numpy as np

def bootstrap_rmse_ci(y_true, y_pred, n_boot=200, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    boot_rmse = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_rmse.append(rmse(y_true[idx], y_pred[idx]))

    low = float(np.percentile(boot_rmse, 100 * alpha / 2))
    high = float(np.percentile(boot_rmse, 100 * (1 - alpha / 2)))
    return low, high