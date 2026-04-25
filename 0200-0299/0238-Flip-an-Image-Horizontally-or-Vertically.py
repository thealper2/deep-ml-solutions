import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    try:
        first_pixel = image[0][0]
        is_3d = isinstance(first_pixel, list)

        height = len(image)
        width = len(image[0])

        for row in image:
            if not isinstance(row, list) or len(row) != width:
                return -1

            if is_3d:
                for pixel in row:
                    if not isinstance(pixel, list) or len(pixel) != 3:
                        return -1

                    if not all(isinstance(ch, (int, float)) for ch in pixel):
                        return -1

            else:
                if not all(isinstance(pixel, (int, float)) for pixel in row):
                    return -1

    except (IndexError, TypeError):
        return -1

    if direction not in ['horizontal', 'vertical']:
        return -1

    if direction == 'horizontal':
        return [row[::-1] for row in image]
    else:
        return image[::-1]