def assign_layer_types(num_layers: int, num_dense_hca: int) -> list:
    """Return a list of 'HCA'/'CSA' strings per layer."""
    layer_types = []

    for i in range(num_layers):
        if i < num_dense_hca:
            layer_types.append('HCA')
        else:
            offset = i - num_dense_hca
            if offset % 2 == 0:
                layer_types.append('CSA')
            else:
                layer_types.append('HCA')


    return layer_types