def select_round_clients(num_clients, client_fraction, seed):
    torch.manual_seed(seed)
    clients = list(range(num_clients))
    K = max(1, round(client_fraction * num_clients))
    perm = torch.randperm(num_clients)
    selected_clients = perm[:K]
    return torch.sort(selected_clients).values.tolist()
