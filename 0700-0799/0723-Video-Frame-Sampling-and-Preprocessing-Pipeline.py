import numpy as np

def sample_and_preprocess(video, num_frames: int) -> list:
    """
    Uniformly sample num_frames frames from a video and normalize to [0, 1].

    Args:
        video: array-like of shape (T, H, W, C) with values in [0, 255].
        num_frames: number of frames to sample.

    Returns:
        Nested list of shape (num_frames, H, W, C) with float values in [0, 1].
    """
    video = np.array(video)
    T, H, W, C = video.shape

    if num_frames == 1:
        indices = [0]
    else:
        positions = np.linspace(0, T - 1, num_frames)
        indices = [int(round(pos)) for pos in positions]

    sampled_frames = video[indices]
    normalized = sampled_frames / 255.0
    return normalized.tolist()
