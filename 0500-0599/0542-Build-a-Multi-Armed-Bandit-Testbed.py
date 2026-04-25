import numpy as np

def create_bandit_testbed(k: int, num_pulls: int, seed: int = 42) -> tuple:
    """
    Build a k-armed bandit testbed and simulate pulling each arm.
    
    Args:
        k: Number of arms
        num_pulls: Number of times to pull each arm
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (true_values, sample_means, optimal_arm)
    """
    np.random.seed(seed)
    true_values = np.random.randn(k)
    sample_means = []

    for arm in range(k):
        rewards = np.random.randn(num_pulls) + true_values[arm]
        sample_mean = np.mean(rewards)
        sample_means.append(sample_mean)

    optimal_arm = int(np.argmax(true_values))
    true_values = [round(float(v), 4) for v in true_values]
    sample_means = [round(float(v), 4) for v in sample_means]
    return true_values, sample_means, optimal_arm