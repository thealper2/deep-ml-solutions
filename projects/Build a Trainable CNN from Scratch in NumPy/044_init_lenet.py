def init_lenet(in_channels, num_classes, seed=0):
    conv1 = init_conv_layer(
        out_channels=6,
        in_channels=in_channels,
        kernel_size=5,
        seed=seed
    )
    conv2 = init_conv_layer(
        out_channels=16,
        in_channels=6,
        kernel_size=5,
        seed=seed + 1
    )

    flattened_size = 16 * 4 * 4

    fc1 = init_linear_layer(flattened_size, 120, seed = seed + 2)
    fc2 = init_linear_layer(120, num_classes, seed = seed + 3)

    return {
        'conv1': conv1,
        'conv2': conv2,
        'fc1': fc1,
        'fc2': fc2,
    }
