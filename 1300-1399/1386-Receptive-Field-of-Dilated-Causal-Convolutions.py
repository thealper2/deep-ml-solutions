def receptive_field(kernel_sizes, dilations):
    """R = 1 + sum((k - 1) * d) over layers. Returns int."""
    R = 1
    for k, d in zip(kernel_sizes, dilations):
        R += (k - 1) * d

    return R

def wavenet_receptive_field(n_layers, kernel_size, n_blocks=1):
    """n_blocks blocks of dilations 1, 2, 4, ..., 2**(n_layers-1). Returns int."""
    R = 1
    for _ in range(n_blocks):
        for layer in range(n_layers):
            dilation = 2 ** layer
            R += (kernel_size - 1) * dilation

    return R

def causal_padding(kernel_size, dilation):
    """Left padding that keeps the sequence length: (kernel_size - 1) * dilation."""
    return (kernel_size - 1) * dilation