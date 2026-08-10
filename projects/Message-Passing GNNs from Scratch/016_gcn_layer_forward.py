def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    h = gcn_linear_transform(node_features, weight, bias)
    src_hat, dst_hat, norm_weight = gcn_renormalize_adjacency(src, dst, num_nodes)
    messages = h[src_hat] * norm_weight.unsqueeze(-1)
    aggregated = scatter_sum_to_nodes(messages, dst_hat, num_nodes)
    
    if activation is not None:
        aggregated = activation(aggregated)

    return aggregated
