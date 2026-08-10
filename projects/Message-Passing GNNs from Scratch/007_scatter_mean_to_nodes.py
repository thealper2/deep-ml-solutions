def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    sums = scatter_sum_to_nodes(edge_features, dst, num_nodes)
    degrees = compute_node_degrees(torch.tensor([], dtype=torch.long), dst, num_nodes)
    mask = degrees > 0
    result = torch.zeros_like(sums)
    result[mask] = sums[mask] / degrees[mask].unsqueeze(1)
    return result
