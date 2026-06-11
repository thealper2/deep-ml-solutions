import torch

def train_test_split_dataset(features, labels, test_fraction, seed):
    N = len(features)
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(N, generator=generator)
    split_idx = int(N * test_fraction)
    train_indices = shuffled_indices[split_idx:]
    test_indices = shuffled_indices[:split_idx]

    tr_f = features[train_indices]
    tr_l = labels[train_indices]
    te_f = features[test_indices]
    te_l = labels[test_indices]
    
    return tr_f, tr_l, te_f, te_l
