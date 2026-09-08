import copy
import torch
import torch.nn as nn

def fit(model, loaders, epochs=5, lr=0.05, seed=42):
    torch.manual_seed(seed)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0
    }
    
    best_acc = -1.0
    best_state = None
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loaders['train'], loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, loaders['val'], loss_fn)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            history['best_epoch'] = epoch
    
    model.load_state_dict(best_state)
    
    return history