def compute_receptive_field(input_size: int, layers: list) -> dict:
    """
    Compute the receptive field size and output spatial size
    for a 1D convolutional network.
    
    Args:
        input_size: The spatial size of the 1D input
        layers: List of tuples (kernel_size, stride, padding)
                for each convolutional layer in order
    
    Returns:
        Dictionary with:
          'receptive_field': int - number of input positions seen by one output neuron
          'output_size': int - spatial size after all layers
    """
    output_size = input_size
    receptive_field = 1
    stride_product = 1

    for kernel_size, stride, padding in layers:
        output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        receptive_field = receptive_field + (kernel_size - 1) * stride_product 
        stride_product *= stride

    return {
        'receptive_field': receptive_field,
        'output_size': output_size
    }