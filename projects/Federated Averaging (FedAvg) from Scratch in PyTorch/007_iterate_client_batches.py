def iterate_client_batches(client_features, client_labels, batch_size, seed):
    torch.manual_seed(seed)
    n = client_features.shape[0]
    indices = torch.randperm(n)
    shuffled_features = client_features[indices]
    shuffled_labels = client_labels[indices]
    batches = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batches.append((shuffled_features[i:end], shuffled_labels[i:end]))

    return batches
