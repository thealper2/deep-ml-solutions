import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    for col in range(X_clean.shape[1]):
        col_data = X_clean[:, col]
        median_val = np.nanmedian(col_data)
        col_data[np.isnan(col_data)] = median_val

    return X_clean
