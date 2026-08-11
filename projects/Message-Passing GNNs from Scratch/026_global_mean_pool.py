def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1

    summed = scatter_sum_to_nodes(node_features, batch_index, num_graphs)
    ones = torch.ones(node_features.shape[0], 1, dtype=node_features.dtype, device=node_features.device)
    counts = scatter_sum_to_nodes(ones, batch_index, num_graphs)
    graph_features = summed / counts
    return graph_features
