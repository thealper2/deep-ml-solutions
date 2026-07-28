"""
NumPy House Price Regression — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  impute_nan_with_mean ──
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    X_imputed = X.copy()
    for col in range(X.shape[1]):
        col_data = X[:, col]
        mask = ~np.isnan(col_data)
        if np.any(mask):
            mean_val = np.mean(col_data[mask])
            X_imputed[mask == False, col] = mean_val
        else:
            X_imputed[:, col] = 0.0

    return X_imputed

# ── Step 002  compute_iqr_bounds ──
def compute_iqr_bounds(X, k=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return lower, upper

# ── Step 003  clip_columns ──
def clip_columns(X, lower, upper):
    return np.clip(X, lower, upper)

# ── Step 004  make_ratio_feature ──
def make_ratio_feature(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)

# ── Step 005  append_column ──
def append_column(X, col):
    if col.ndim == 1:
        col = col.reshape(-1, 1)

    return np.hstack([X, col])

# ── Step 006  one_hot_encode ──
def one_hot_encode(labels):
    unique = np.unique(labels)
    label_to_idx = {label: i for i, label in enumerate(unique)}
    N = len(labels)
    C = len(unique)
    one_hot = np.zeros((N, C), dtype=float)
    for i, label in enumerate(labels):
        one_hot[i, label_to_idx[label]] = 1.0

    return one_hot

# ── Step 007  fit_standardizer ──
def fit_standardizer(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)
    std[std == 0] = 1.0
    return mean, std

# ── Step 008  apply_standardizer ──
def apply_standardizer(X, mean, std):
    return (X - mean) / std

# ── Step 009  add_bias_column ──
def add_bias_column(X):
    N = X.shape[0]
    ones = np.ones((N, 1), dtype=X.dtype)
    return np.hstack([ones, X])

# ── Step 010  make_shuffled_indices ──
def make_shuffled_indices(n_samples, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return indices

# ── Step 011  partition_indices ──
def partition_indices(indices, train_ratio, val_ratio):
    N = len(indices)
    test_ratio = 1 - train_ratio - val_ratio
    train_size = int(train_ratio * N)
    val_size = int(val_ratio * N)
    test_size = int(test_ratio * N)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]
    return train_idx, val_idx, test_idx

# ── Step 012  subset_xy ──
def subset_xy(X, y, indices):
    return X[indices], y[indices]

# ── Step 013  ols_fit ──
def ols_fit(X, y):
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return theta

# ── Step 014  ols_predict ──
def ols_predict(X, theta):
    return X @ theta

# ── Step 015  mean_absolute_error ──
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ── Step 016  root_mean_squared_error ──
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ── Step 017  r_squared ──
def r_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    r_s = 1 - (ss_res / ss_tot)
    return r_s if not np.isnan(r_s) else 0.0

# ── Step 018  residual_summary ──
def residual_summary(y_true, y_pred):
    residuals = y_true - y_pred
    mean = float(np.mean(residuals))
    std = float(np.std(residuals))
    median_abs = float(np.median(np.abs(residuals)))
    return {'mean': mean, 'std': std, 'median_abs': median_abs}

# ── Step 019  prepare_cleaned_features ──
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X_imputed = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_imputed, k=iqr_k)
    X_clipped = clip_columns(X_imputed, lower, upper)
    return X_clipped

# ── Step 020  assemble_feature_matrix ──
import numpy as np

def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    numerator = X_num[:, ratio_num_idx]
    denominator = X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(numerator, denominator)
    X_extended = append_column(X_num, ratio)

    if cat_labels is not None:
        cat_encoded = one_hot_encode(cat_labels)
        X_extended = np.hstack([X_extended, cat_encoded])

    return X_extended

# ── Step 021  make_train_val_test ──
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    np.random.seed(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx]
    }

# ── Step 022  standardize_and_add_bias ──
def standardize_and_add_bias(splits):
    X_train = splits['X_train']
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0, ddof=0)
    std[std == 0] = 1.0

    std_splits = {}
    for key, value in splits.items():
        if key.startswith('X_'):
            X_std = (value - mean) / std
            std_splits[key] = add_bias_column(X_std)
        else:
            std_splits[key] = value

    return std_splits, mean, std

# ── Step 023  evaluate_predictions ──
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

# ── Step 024  house_price_pipeline ──
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)
    X_full = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels)
    splits = make_train_val_test(X_full, y, train_ratio, val_ratio, seed)
    std_splits, mean, std = standardize_and_add_bias(splits)
    theta = ols_fit(std_splits['X_train'], std_splits['y_train'])
    y_val_pred = ols_predict(std_splits['X_val'], theta)
    y_test_pred = ols_predict(std_splits['X_test'], theta)
    val_metrics = evaluate_predictions(std_splits['y_val'], y_val_pred)
    test_metrics = evaluate_predictions(std_splits['y_test'], y_test_pred)
    
    return {
        'theta': theta,
        'y_test': std_splits['y_test'],
        'y_test_pred': y_test_pred,
        'test_metrics': test_metrics,
        'val_metrics': val_metrics
    }

# ── Scaffold (runner) ──
"""Demo: end-to-end NumPy house-price OLS regression pipeline."""
import numpy as np


def main():
    np.random.seed(0)
    n = 200
    # Synthetic tabular features: [rooms, households, age, income]
    rooms = np.random.uniform(2.0, 8.0, size=n)
    households = np.random.uniform(1.0, 5.0, size=n)
    age = np.random.uniform(5.0, 50.0, size=n)
    income = np.random.uniform(1.0, 10.0, size=n)
    X = np.column_stack([rooms, households, age, income])
    # Inject a few NaNs and outliers
    X[5, 0] = np.nan
    X[12, 3] = np.nan
    X[20, 2] = 200.0
    # Categorical district labels
    cat_labels = np.random.choice(["A", "B", "C"], size=n)
    # Target: noisy linear function of rooms/households and income
    y = 50.0 + 30.0 * (rooms / (households + 1e-8)) + 15.0 * income
    y = y + np.random.normal(0.0, 5.0, size=n)
    y[20] = y[20] + 100.0

    metrics, y_test, y_pred, *_ = house_price_pipeline(
        X,
        y,
        ratio_num_idx=0,
        ratio_den_idx=1,
        cat_labels=cat_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
        iqr_k=1.5,
    )

    print("Test metrics:")
    print("  MAE :", round(float(metrics["mae"]), 4))
    print("  RMSE:", round(float(metrics["rmse"]), 4))
    print("  R^2 :", round(float(metrics["r2"]), 4))
    print("Residual summary:", metrics["residuals"])
    print("y_test[:5]:", np.round(y_test[:5], 3))
    print("y_pred[:5]:", np.round(y_pred[:5], 3))


if __name__ == "__main__":
    main()
