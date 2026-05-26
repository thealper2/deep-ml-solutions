import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    result = data.copy()
    if strategy == 'mean':
        col_means = np.nanmean(data, axis=0)
        inds = np.where(np.isnan(result))
        result[inds] = np.take(col_means, inds[1])
        return result.tolist()
    elif strategy == 'median':
        col_medians = np.nanmedian(data, axis=0)
        inds = np.where(np.isnan(result))
        result[inds] = np.take(col_medians, inds[1])
        return result.tolist()
    elif strategy == 'mode':
        for col in range(data.shape[1]):
            col_data = data[:, col]
            non_nan = col_data[~np.isnan(col_data)]
            if len(non_nan) > 0:
                values, counts = np.unique(non_nan, return_counts=True)
                mode = values[np.argmax(counts)]
                inds = np.isnan(result[:, col])
                result[inds, col] = mode
                
        return result.tolist()
    else:
        raise ValueError('Unknown strategy')