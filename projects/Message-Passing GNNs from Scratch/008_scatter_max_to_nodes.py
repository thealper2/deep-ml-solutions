def scatter_max_to_nodes(edge_features, dst, num_nodes):
    E, F = edge_features.shape
    device = edge_features.device
    dtype = edge_features.dtype
    
    if E == 0:
        return torch.full((num_nodes, F), float('-inf'), dtype=dtype, device=device)
    
    result = torch.full((num_nodes, F), float('-inf'), dtype=dtype, device=device)
    dst_expanded = dst.unsqueeze(1).expand(-1, F)
    result.scatter_reduce_(
        dim=0,
        index=dst_expanded,
        src=edge_features,
        reduce='amax',
        include_self=False
    )
    
    return result
