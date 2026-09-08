import numpy as np
import torch

def random_search(loaders, n_trials=4, epochs=2, seed=42):
    rng = np.random.default_rng(seed)
    
    hidden1_options = [100, 200, 300]
    hidden2_options = [50, 100]
    lr_options = [0.01, 0.05, 0.1]
    
    trials = []
    best_trial = None
    best_acc = -1.0
    
    for _ in range(n_trials):
        hidden1 = rng.choice(hidden1_options)
        hidden2 = rng.choice(hidden2_options)
        lr = rng.choice(lr_options)
        
        torch.manual_seed(seed)
        model = MLP(hidden1, hidden2)
        
        history = fit(model, loaders, epochs=epochs, lr=lr, seed=seed)
        val_acc = max(history['val_acc'])
        
        trial = {
            'hidden1': int(hidden1),
            'hidden2': int(hidden2),
            'lr': float(lr),
            'val_acc': float(val_acc)
        }
        trials.append(trial)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_trial = trial
    
    return {
        'trials': trials,
        'best': best_trial
    }