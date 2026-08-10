def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    if edge_weight is None:
        ones = torch.ones_like(dst, dtype=torch.float)
        degrees = torch.zeros(num_nodes, dtype=torch.float, device=dst.device)
        degrees.scatter_add_(0, dst, ones)
    else:
        degrees = torch.zeros(num_nodes, dtype=torch.float, device=dst.device)
        degrees.scatter_add_(0, dst, edge_weight.float())

    return degrees
