def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    
    if activations is None:
        activations = [None] * len(param_list)
    
    embeddings = node_features
    all_layer_outputs = []
    
    for i, params in enumerate(param_list):
        weight = params['weight']
        bias = params.get('bias', None)
        activation = activations[i] if i < len(activations) else None
        
        embeddings = gcn_layer_forward(
            embeddings, src, dst, weight, bias, num_nodes, activation
        )
        all_layer_outputs.append(embeddings)
    
    return embeddings, all_layer_outputs
