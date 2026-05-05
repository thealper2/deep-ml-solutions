import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, size=(runs, n))
        pop_mean = 0.5
        pop_std = np.sqrt(1/12)
    elif distribution == 'exponential':
        samples = np.random.exponential(1.0, size=(runs, n))
        pop_mean = 1.0
        pop_std = 1.0
    elif distribution == 'bernoulli':
        p = 0.3
        samples = (np.random.rand(runs, n) < p).astype(float)
        pop_mean = p
        pop_std = np.sqrt(p * (1 - p))
    else:
        raise ValueError("Invalid distribution")

    sample_means = np.mean(samples, axis=1)
    std_error = pop_std / np.sqrt(n)
    z_scores = (sample_means - pop_mean) / std_error

    return {
        'mean': round(float(np.mean(z_scores)), 3),
        'std': round(float(np.std(z_scores)), 3)
    }