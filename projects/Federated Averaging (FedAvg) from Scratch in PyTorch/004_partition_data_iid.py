def partition_data_iid(train_features, train_labels, num_clients, seed):
    if num_clients == 0:
        num_clients = 1

    torch.manual_seed(seed)
    M = train_features.shape[0]
    indices = torch.randperm(M)
    shuffled_features = train_features[indices]
    shuffled_labels = train_labels[indices]
    client_data = []
    chunk_size = M // num_clients
    remainder = M % num_clients
    start = 0
    for i in range(num_clients):
        end = start + chunk_size + (1 if i < remainder else 0)
        client_data.append((shuffled_features[start:end], shuffled_labels[start:end]))
        start = end

    return client_data
