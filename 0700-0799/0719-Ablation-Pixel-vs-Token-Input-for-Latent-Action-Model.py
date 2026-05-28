import numpy as np

def compare_input_modalities(prev_frames: np.ndarray, next_frames: np.ndarray, codebook: np.ndarray) -> dict:
    """
    Compare pixel-input vs token-input latent action reconstruction.
    Returns a dict with keys: 'pixel_mse', 'token_mse', 'winner'.
    """
    pixel_action = np.round(next_frames - prev_frames)
    pixel_reconstructed = prev_frames + pixel_action
    pixel_mse = np.mean((pixel_reconstructed - next_frames) ** 2)

    K, D = codebook.shape
    N = prev_frames.shape[0]

    prev_tokens = np.zeros_like(prev_frames)
    next_tokens = np.zeros_like(next_frames)

    for i in range(N):
        dist_prev = np.linalg.norm(codebook - prev_frames[i], axis=1)
        prev_idx = np.argmin(dist_prev)
        prev_tokens[i] = codebook[prev_idx]

        dist_next = np.linalg.norm(codebook - next_frames[i], axis=1)
        next_idx = np.argmin(dist_next)
        next_tokens[i] = codebook[next_idx]

    token_action = np.round(next_tokens - prev_tokens)
    token_reconstructed = prev_tokens + token_action
    token_mse = np.mean((token_reconstructed - next_frames) ** 2)

    if pixel_mse < token_mse:
        winner = 'pixel'
    elif pixel_mse > token_mse:
        winner = 'token'
    else:
        winner = 'tie'

    return {
        'pixel_mse': round(pixel_mse, 4),
        'token_mse': round(token_mse, 4),
        'winner': winner,
    }