def build_graph_regression_dataset(num_graphs, num_nodes_range, num_node_features, edge_prob=0.3, seed=0):
    lo, hi = num_nodes_range
    graphs = []
    
    for i in range(num_graphs):
        num_nodes = lo + (i % (hi - lo + 1))
        graph = generate_molecule_like_graph(
            num_nodes, num_node_features, edge_prob=edge_prob, seed=seed + i
        )
        graphs.append(graph)
    
    return graphs
