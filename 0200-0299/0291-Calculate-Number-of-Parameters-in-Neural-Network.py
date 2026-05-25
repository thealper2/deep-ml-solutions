def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    total = 0
    for layer in layers:
        if layer['type'] == 'dense':
            total += layer['input_size'] * layer['output_size']
            if layer.get('use_bias', True):
                total += layer['output_size']

        elif layer['type'] == 'conv2d':
            in_channels = layer['in_channels']
            out_channels = layer['out_channels']
            kernel_size = layer['kernel_size']

            if isinstance(kernel_size, int):
                kernel_h = kernel_w = kernel_size
            else:
                kernel_h, kernel_w = kernel_size

            total += out_channels * in_channels * kernel_h * kernel_w
            if layer.get('use_bias', True):
                total += out_channels

        elif layer['type'] == 'embedding':
            total += layer['num_embeddings'] * layer['embedding_dim']

    return total