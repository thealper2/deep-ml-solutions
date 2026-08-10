def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    if edge_weight is None:
        degrees = compute_node_degrees(src, dst, num_nodes)
    else:
        degrees = compute_node_degrees(src, dst, num_nodes, edge_weight)

    inv_sqrt_deg = torch.zeros(num_nodes, dtype=torch.float, device=src.device)
    mask = degrees > 0
    inv_sqrt_deg[mask] = 1.0 / torch.sqrt(degrees[mask])

    inv_sqrt_src = inv_sqrt_deg[src]
    inv_sqrt_dst = inv_sqrt_deg[dst]

    if edge_weight is None:
        normalized_weights = inv_sqrt_src * inv_sqrt_dst
    else:
        normalized_weights = edge_weight.float() * inv_sqrt_src * inv_sqrt_dst

    return normalized_weights
