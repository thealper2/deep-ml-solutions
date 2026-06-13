import numpy as np

def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, len(data) - block_size, size=batch_size)
    x = np.array([data[i:i+block_size] for i in offsets])
    y = np.array([data[i+1:i+1+block_size] for i in offsets])
    return np.array([x, y])
