import numpy as np

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.
    
    Args:
        image: 2D list/array representing a grayscale image
               with values in range [0, 255]
    
    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """
    if not isinstance(image, (list, np.ndarray)):
        return -1
    
    image = np.array(image, dtype=np.float32)
    
    if image.ndim != 2:
        return -1
    
    H, W = image.shape
    
    if H < 3 or W < 3:
        return -1
    
    if np.any(image < 0) or np.any(image > 255):
        return -1
    
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    out_h = H - 2
    out_w = W - 2
    
    Gx = np.zeros((out_h, out_w))
    Gy = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            window = image[i:i+3, j:j+3]
            Gx[i, j] = np.sum(window * Kx)
            Gy[i, j] = np.sum(window * Ky)
    
    magnitude = np.hypot(Gx, Gy)
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max()) * 255
    
    return magnitude.astype(np.uint8).tolist()
