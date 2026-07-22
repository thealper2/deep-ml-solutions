import math

def rotation_layer(X, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = []
    for x, y in X:
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        rotated.append([new_x, new_y])

    return rotated
