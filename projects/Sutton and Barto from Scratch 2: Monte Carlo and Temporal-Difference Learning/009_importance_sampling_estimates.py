import numpy as np

def importance_sampling_estimates(returns, ratios):
    returns = np.asarray(returns, dtype=float)
    ratios = np.asarray(ratios, dtype=float)

    weighted_returns = ratios * returns

    cumulative_weighted_returns = np.cumsum(weighted_returns)
    cumulative_ratios = np.cumsum(ratios)

    n = len(returns)

    ordinary = cumulative_weighted_returns / np.arange(1, n + 1)

    weighted = np.zeros(n, dtype=float)

    np.divide(
        cumulative_weighted_returns,
        cumulative_ratios,
        out=weighted,a
        where=cumulative_ratios != 0
    )

    return ordinary, weighted