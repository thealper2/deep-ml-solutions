import numpy as np

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray, max_pixel_value: float) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio between two images.

    Args:
        original: numpy array of the original image
        reconstructed: numpy array of the reconstructed image (same shape)
        max_pixel_value: maximum possible pixel value

    Returns:
        PSNR value in decibels (float). Returns float('inf') if images are identical.
    """
    mean_squared_error = lambda x, y: np.mean((x - y) ** 2)
    mse = mean_squared_error(original, reconstructed)
    if mse == 0:
        return float('inf')

    psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse)
    return round(psnr_value, 4)