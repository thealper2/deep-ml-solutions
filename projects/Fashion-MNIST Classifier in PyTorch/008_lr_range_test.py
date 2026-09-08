import torch
import torch.nn as nn

def lr_range_test(make_model, loader, lrs, n_batches=20, seed=42):
    results = {}
    
    for lr in lrs:
        torch.manual_seed(seed)
        model = make_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        total_loss = 0.0
        num_batches = 0
        
        for i, (xb, yb) in enumerate(loader):
            if i >= n_batches:
                break
            
            model.train()
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        results[lr] = total_loss / num_batches
    
    return results