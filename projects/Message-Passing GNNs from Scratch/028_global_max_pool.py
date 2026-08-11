def global_max_pool(node_features, batch_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
    
    graph_feats = scatter_max_to_nodes(node_features, batch_index, num_graphs)
    
    return graph_feats
