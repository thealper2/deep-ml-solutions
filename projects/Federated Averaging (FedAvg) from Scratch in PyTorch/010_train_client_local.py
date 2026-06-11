def train_client_local(model, client_features, client_labels, local_epochs, batch_size, learning_rate, seed):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(local_epochs):
        batches = iterate_client_batches(client_features, client_labels, batch_size, seed + epoch)
        for batch_features, batch_labels in batches:
            local_sgd_step(model, optimizer, batch_features, batch_labels)

    return model.state_dict()
