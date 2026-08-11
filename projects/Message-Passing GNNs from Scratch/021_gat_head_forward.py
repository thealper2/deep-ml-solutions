def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    logits, transformed = gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight)
    attn_coeffs = gat_masked_neighbor_softmax(logits, dst, num_nodes)
    messages = transformed[src] * attn_coeffs.unsqueeze(-1)
    head_out = scatter_sum_to_nodes(messages, dst, num_nodes)

    if bias is not None:
        head_out = head_out + bias

    if activation is not None:
        head_out = activation(head_out)

    return head_out, attn_coeffs
