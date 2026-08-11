def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    head_outputs = []
    all_attn = []

    for params in head_params:
        weight = params['weight']
        attn_src = params['attn_src']
        attn_dst = params['attn_dst']
        bias = params.get('bias', None)

        head_out, attn_coeffs = gat_head_forward(
            node_features, src, dst, weight, attn_src, attn_dst,
            bias=bias, num_nodes=num_nodes, activation=None
        )
        head_outputs.append(head_out)
        all_attn.append(attn_coeffs)

    out = merge_gat_heads(head_outputs, mode=merge_mode)

    if activation is not None:
        out = activation(out)

    return out, all_attn
