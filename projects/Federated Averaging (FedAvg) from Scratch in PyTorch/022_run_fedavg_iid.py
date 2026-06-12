def run_fedavg_iid(train_features, train_labels, test_features, test_labels, model_config, num_clients, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    client_partitions = partition_data_iid(train_features, train_labels, num_clients, seed)
    model, accuracies = run_fedavg(
        client_partitions, test_features, test_labels, model_config,
        num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed
    )
    return accuracies
