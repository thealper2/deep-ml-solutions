import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.
    
    Args:
        image: RGB image of shape (H, W, 3) with values in [0, 255]
        alpha: Array of 3 random coefficients for principal components
    
    Returns:
        Augmented image of shape (H, W, 3) with values clamped to [0, 255]
    """
    img = image.astype(np.float64)
    pixels = img.reshape(-1, 3)

    cov = np.cov(pixels, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    delta = eigvecs @ (alpha * np.sqrt(eigvals))
    return np.clip(img + delta, 0, 255)
