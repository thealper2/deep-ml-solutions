def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    N, F = num_nodes, edge_features.shape[1]
    result = torch.zeros((N, F), dtype=edge_features.dtype, device=edge_features.device)
    result.scatter_add_(0, dst.unsqueeze(1).expand(-1, F), edge_features)
    return result
