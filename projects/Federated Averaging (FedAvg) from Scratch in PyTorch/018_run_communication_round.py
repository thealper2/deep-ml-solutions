def run_communication_round(global_state, client_partitions, selected_clients, model_config, local_epochs, batch_size, learning_rate, seed):
    client_states = []
    client_sample_counts = []

    for client_idx in selected_clients:
        client_features, client_labels = client_partitions[client_idx]
        model = build_mlp_classifier(
            model_config['input_size'],
            model_config['hidden_size'],
            model_config['num_classes'],
        )
        load_model_state(model, global_state)
        trained_state = train_client_local(
            model, client_features, client_labels,
            local_epochs, batch_size, learning_rate,
            seed + client_idx,
        )
        client_states.append(trained_state)
        client_sample_counts.append(client_features.shape[0])

    new_global_state = aggregate_weighted_average(client_states, client_sample_counts)
    return new_global_state
