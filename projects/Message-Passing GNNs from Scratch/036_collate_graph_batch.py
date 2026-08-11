def collate_graph_batch(graphs):
    x_list = []
    edge_list = []
    batch_list = []
    y_list = []
    
    offset = 0
    
    for i, g in enumerate(graphs):
        x = g['x']
        n = x.shape[0]
        x_list.append(x)
        edge_index = g['edge_index'] + offset
        edge_list.append(edge_index)
        batch = torch.full((n,), i, dtype=torch.long)
        batch_list.append(batch)
        y = torch.tensor(g['y'], dtype=torch.float32)
        y_list.append(y)
        
        offset += n
    
    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = torch.cat(edge_list, dim=1)
    batch_batch = torch.cat(batch_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    
    return {
        'x': x_batch,
        'edge_index': edge_index_batch,
        'batch': batch_batch,
        'y': y_batch
    }
