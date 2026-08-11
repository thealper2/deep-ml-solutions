def train_graph_regressor(params, graphs, forward_fn, num_epochs, lr, batch_size=8):
    """Train a graph regressor over multiple epochs of mini-batches.

    Args:
        params: dict of trainable torch tensors.
        graphs: list of graph dicts with keys x, edge_index, y.
        forward_fn: callable(params, batch) -> predictions.
        num_epochs: number of training epochs.
        lr: learning rate for SGD updates.
        batch_size: graphs per mini-batch (default 8).

    Returns:
        history: dict with 'loss' and 'mae' lists of per-epoch floats.
        params: updated parameter dict.
    """
    n = len(graphs)
    history = {'loss': [], 'mae': []}
    
    for epoch in range(num_epochs):
        indices = torch.randperm(n).tolist()
        shuffled_graphs = [graphs[i] for i in indices]
        
        total_loss = 0.0
        n_batches = 0
        
        for start in range(0, n, batch_size):
            batch_graphs = shuffled_graphs[start:start + batch_size]
            batch = collate_graph_batch(batch_graphs)
            
            result = gnn_train_step(params, batch, forward_fn, mse_loss, lr)
            total_loss += result['loss']
            params = result['params']
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        with torch.no_grad():
            full_batch = collate_graph_batch(graphs)
            predictions = forward_fn(params, full_batch)
            mae = mae_metric(predictions, full_batch['y'])
        
        history['loss'].append(avg_loss)
        history['mae'].append(mae)
    
    return history, params
