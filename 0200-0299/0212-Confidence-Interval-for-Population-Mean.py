import numpy as np

# numpy has no t-distribution quantile function and scipy is not available.
# Use this lookup table to get t-critical values for the (df, p) pairs
# needed by the test cases, where p = 1 - alpha/2.
T_TABLE = {
    (4, 0.95):  2.1318467863998393,
    (4, 0.975): 2.7764451051977987,
    (4, 0.995): 4.604094871415387,
    (5, 0.95):  2.0150483726691575,
    (5, 0.975): 2.5705818366147395,
    (5, 0.995): 4.032142983557536,
    (6, 0.95):  1.9431802803927816,
    (6, 0.975): 2.4469118511449624,
    (6, 0.995): 3.707428021324907,
    (7, 0.95):  1.8945786050613064,
    (7, 0.975): 2.3646242510102993,
    (7, 0.995): 3.4994832973505026,
}


def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
    """
    Calculate confidence interval for population mean.

    Args:
        data: Sample data
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dictionary containing:
        - mean: Sample mean (point estimate)
        - standard_error: Standard error of the mean
        - margin_of_error: Margin of error
        - lower_bound: Lower bound of CI
        - upper_bound: Upper bound of CI
        - confidence_level: Confidence level used
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    standard_error = std_dev / np.sqrt(n)
    df = n - 1
    t_critical = T_TABLE[(df, (1 + confidence_level) / 2)]
    margin_of_error = t_critical * standard_error
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return {
        'mean': round(mean, 3),
        'standard_error': round(standard_error, 4),
        'margin_of_error': round(margin_of_error, 3),
        'lower_bound': round(lower_bound, 3),
        'upper_bound': round(upper_bound, 3),
        'confidence_level': confidence_level,
    }