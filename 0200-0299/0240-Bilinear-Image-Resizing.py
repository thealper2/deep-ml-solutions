import numpy as np

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    """
    Resize an image using bilinear interpolation.
    
    Args:
        image: 2D (grayscale) or 3D (RGB) array representing an image
        new_height: Target height of the resized image
        new_width: Target width of the resized image
    
    Returns:
        Resized image as a nested list with values rounded to 2 decimal places
    """
    image = np.asarray(image, dtype=float)

    if image.ndim == 2:
        image = image[..., None]
        grayscale = True
    else:
        grayscale = False
    
    h, w, c = image.shape
    out = np.empty((new_height, new_width, c), dtype=float)

    y_scale = h / new_height
    x_scale = w / new_width

    for i in range(new_height):
        sy = i * y_scale
        y0 = min(int(np.floor(sy)), h - 1)
        y1 = min(y0 + 1, h - 1)
        wy = sy - y0

        for j in range(new_width):
            sx = j * x_scale
            x0 = min(int(np.floor(sx)), w - 1)
            x1 = min(x0 + 1, w - 1)
            wx = sx - x0

            top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
            bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
            out[i, j] = (1 - wy) * top + wy * bottom

    out = np.round(out, 2)

    if grayscale:
        return out[..., 0].tolist()

    return out.tolist()
