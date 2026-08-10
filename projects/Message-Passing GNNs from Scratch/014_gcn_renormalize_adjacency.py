def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    src_hat, dst_hat = add_self_loops(src, dst, num_nodes)
    norm_weight = symmetric_normalize_edge_weights(src_hat, dst_hat, num_nodes)
    return src_hat, dst_hat, norm_weight
