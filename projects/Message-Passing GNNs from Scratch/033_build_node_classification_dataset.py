def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    graphs = []
    
    for i in range(num_graphs):
        if seed is not None:
            g_seed = seed + i
        else:
            g_seed = None
        
        graph = generate_sbm_graph(
            num_nodes, num_classes, p_in, p_out, feature_dim, seed=g_seed
        )
        graphs.append(graph)
    
    return graphs
