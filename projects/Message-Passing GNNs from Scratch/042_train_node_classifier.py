def train_node_classifier(params, dataset, forward_fn, num_epochs, lr, mask_key='train_mask'):
    x = dataset['x']
    edge_index = dataset['edge_index']
    y = dataset['y']
    mask = dataset[mask_key]
    
    x_masked = x[mask]
    y_masked = y[mask]
    
    def wrapped_forward(params, batch):
        logits_full = forward_fn(params, batch['x'], batch['edge_index'])
        logits_masked = logits_full[batch['mask']]
        return logits_masked
    
    batch = {
        'x': x,
        'edge_index': edge_index,
        'mask': mask,
        'y': y_masked
    }
    
    history = []
    
    for _ in range(num_epochs):
        result = gnn_train_step(params, batch, wrapped_forward, cross_entropy_loss, lr)
        step_loss = result['loss']
        params = result['params']
        
        with torch.no_grad():
            logits_full = forward_fn(params, x, edge_index)
            logits_masked = logits_full[mask]
            acc = accuracy_metric(logits_masked, y_masked)
        
        history.append({'loss': step_loss, 'accuracy': acc})
    
    return {'history': history, 'params': params}
