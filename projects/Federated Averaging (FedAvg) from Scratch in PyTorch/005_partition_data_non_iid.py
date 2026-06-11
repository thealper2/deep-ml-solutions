import torch

def partition_data_non_iid(train_features, train_labels, num_clients, shards_per_client, seed):
    torch.manual_seed(seed)
    
    unique_labels = torch.unique(train_labels)
    num_labels = len(unique_labels)
    
    label_to_indices = {}
    for label in unique_labels:
        label_to_indices[int(label.item())] = torch.where(train_labels == label)[0]
    
    shuffled_labels = unique_labels[torch.randperm(num_labels)]
    
    client_data = []
    shard_idx = 0
    total_shards_needed = num_clients * shards_per_client
    
    label_list = shuffled_labels.tolist()
    while len(label_list) < total_shards_needed:
        label_list.extend(shuffled_labels.tolist())
    label_list = label_list[:total_shards_needed]
    
    for client in range(num_clients):
        client_shards = label_list[client * shards_per_client:(client + 1) * shards_per_client]
        
        client_indices = []
        for label in client_shards:
            client_indices.extend(label_to_indices[int(label)].tolist())
        
        client_indices_tensor = torch.tensor(client_indices, dtype=torch.long)
        perm = torch.randperm(len(client_indices_tensor))
        client_indices_tensor = client_indices_tensor[perm]
        
        client_data.append((train_features[client_indices_tensor], train_labels[client_indices_tensor]))
    
    return client_data
