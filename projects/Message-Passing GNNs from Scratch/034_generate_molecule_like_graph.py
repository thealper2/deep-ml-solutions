def generate_molecule_like_graph(num_nodes, num_node_features, edge_prob=0.3, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, num_node_features)
    src_list = []
    dst_list = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1).item() < edge_prob:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)
    
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    node_means = x.mean(dim=1)
    y = (deg * node_means).mean()
    
    return {
        'x': x,
        'edge_index': edge_index,
        'y': y
    }
