import math

def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    Calculate the PMF of the hypergeometric distribution.
    
    Args:
        N: Total population size
        K: Number of success states in population
        n: Number of draws (without replacement)
        k: Number of observed successes
    
    Returns:
        float: P(X = k), rounded to 4 decimal places
    """
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    
    comb = lambda a, b: 0 if b < 0 or b > a else math.comb(a, b)
    prob = comb(K, k) * comb(N - K, n - k) / comb(N, n)
    return round(prob, 4)