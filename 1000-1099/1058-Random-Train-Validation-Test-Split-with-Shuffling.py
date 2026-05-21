import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n = len(data)
    indices = np.random.default_rng(seed).permutation(n)
    data = data[indices]
    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)
    train, val, test = np.split(data, [train_end, validation_end])
    return [train, val, test]