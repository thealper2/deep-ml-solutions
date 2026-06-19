def pad_2d(images, pad):
    if pad == 0:
        return images

    return np.pad(
        images,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant',
        constant_values=0
    )
