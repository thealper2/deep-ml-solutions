def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start = c * num_nodes // num_classes
        end = (c + 1) * num_nodes // num_classes
        node_labels[start:end] = c

    node_features = torch.randn(num_nodes, feature_dim)

    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if node_labels[i] == node_labels[j]:
                p = p_in
            else:
                p = p_out

            if torch.rand(1).item() < p:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'node_labels': node_labels,
        'num_nodes': num_nodes,
    }
