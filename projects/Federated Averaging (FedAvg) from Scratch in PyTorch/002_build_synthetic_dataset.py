import torch

def build_synthetic_dataset(num_samples, input_size, num_classes, seed):
    torch.manual_seed(seed)
    features = torch.randn(num_samples, input_size).float()
    labels = torch.randint(0, num_classes, (num_samples,)).long()
    return features, labels
