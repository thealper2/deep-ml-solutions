def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    L = len(layer_param_list)

    if merge_modes is None:
        merge_modes = ['concat'] * L

    if activations is None:
        activations = [None] * L

    x = node_features
    all_layer_outputs = []

    for i in range(L):
        x, _ = gat_layer_forward(
            x, src, dst,
            head_params=layer_param_list[i],
            merge_mode=merge_modes[i],
            num_nodes=num_nodes,
            activation=activations[i]
        )
        all_layer_outputs.append(x)

    return x, all_layer_outputs
