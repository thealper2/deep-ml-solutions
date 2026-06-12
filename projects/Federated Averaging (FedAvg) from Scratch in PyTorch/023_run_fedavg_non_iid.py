def run_fedavg_non_iid(train_features, train_labels, test_features, test_labels, model_config, num_clients, shards_per_client, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    client_partitions = partition_data_non_iid(train_features, train_labels, num_clients, shards_per_client, seed)
    model, accuracies = run_fedavg(
        client_partitions, test_features, test_labels, model_config,
        num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed
    )
    return model, accuracies
