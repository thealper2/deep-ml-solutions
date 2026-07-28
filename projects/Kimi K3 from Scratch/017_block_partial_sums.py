def block_partial_sums(layer_outputs):
    """Running sums of a block's layer outputs; entry i sums outputs 0..i.

    Returns a list of independent (T, d) arrays; last entry = block sum b_n.
    """
    running = []
    cumsum = np.zeros_like(layer_outputs[0])
    for out in layer_outputs:
        cumsum = cumsum + out
        running.append(cumsum.copy())

    return running
