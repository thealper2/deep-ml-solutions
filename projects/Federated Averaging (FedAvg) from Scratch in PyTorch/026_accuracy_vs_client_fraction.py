def accuracy_vs_client_fraction(client_partitions, test_features, test_labels, model_config, client_fraction_list, num_rounds, local_epochs, batch_size, learning_rate, seed):
    result = {}
    for fraction in client_fraction_list:
        _, accuracies = run_fedavg(
            client_partitions, test_features, test_labels, model_config,
            num_rounds, fraction, local_epochs, batch_size, learning_rate, seed
        )

        result[float(fraction)] = accuracies[-1]

    return result
