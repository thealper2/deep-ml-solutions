import numpy as np

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or numpy array of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """
    if not isinstance(img, list) or not img:
        return -1

    if not isinstance(img[0], list):
        return -1

    rows = len(img)
    cols = len(img[0])

    if rows == 0 or cols == 0:
        return -1

    for row in img:
        if not isinstance(row, list) or len(row) != cols:
            return -1

        if not all(isinstance(pixel, int) for pixel in row):
            return -1

    if not isinstance(pad_width, int) or pad_width < 0:
        return -1

    if pad_width == 0:
        return [row[:] for row in img]

    new_rows = rows + 2 * pad_width
    new_cols = cols + 2 * pad_width

    zero_row = [0] * new_cols
    padded = [zero_row[:] for _ in range(new_rows)]

    for i in range(rows):
        for j in range(cols):
            padded[i + pad_width][j + pad_width] = img[i][j]

    return padded