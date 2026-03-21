import numpy as np

def first_frame_anchor_inject(
    history_frames: np.ndarray,
    first_frame: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Prepend a lightly noised first frame anchor to the history.

    Args:
        history_frames: array of shape (T, H, W, C)
        first_frame:    array of shape (H, W, C)
        sigma_min:      minimum noise std
        sigma_max:      maximum noise std
        rng:            seeded numpy random generator

    Returns:
        Array of shape (T+1, H, W, C)
    """
    sigma = rng.uniform(sigma_min, sigma_max)
    noise = rng.normal(loc=0.0, scale=sigma, size=first_frame.shape)
    noisy_anchor = first_frame + noise
    result = np.concatenate([noisy_anchor[np.newaxis, ...], history_frames], axis=0)
    return result
