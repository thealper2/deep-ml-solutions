import torch

def save_model(model, path):
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'hidden1': model.fc1.out_features,
            'hidden2': model.fc2.out_features,
            'n_classes': model.out.out_features
        }
    }, path)

def load_model(path):
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint['config']
    model = MLP(
        hidden1=config['hidden1'],
        hidden2=config['hidden2'],
        n_classes=config['n_classes']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model