def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=num_samples)
    x = rng.standard_normal((num_samples, in_channels, image_size, image_size))
    shift = (num_classes - 1) / 2
    for k in range(num_classes):
        mask = y == k
        if np.any(mask):
            x[mask] = x[mask] + (k - shift)

    return x, y
