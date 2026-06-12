def rounds_to_target_vs_local_epochs(client_partitions, test_features, test_labels, model_config, local_epochs_list, target_accuracy, num_rounds, client_fraction, batch_size, learning_rate, seed):
    result = {}

    for local_epochs in local_epochs_list:
        _, accuracies = run_fedavg(
            client_partitions, test_features, test_labels, model_config,
            num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed
        )

        found_round = None
        for round_idx, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                found_round = round_idx
                break

        result[local_epochs] = found_round

    return result
