import torch
import numpy as np

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def predict_classes(model, images):
    x = torch.tensor(images, dtype=torch.float32) / 255.0
    
    mean = 0.2860
    std = 0.3530
    x = (x - mean) / std
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
    
    return [CLASS_NAMES[idx] for idx in preds.tolist()]