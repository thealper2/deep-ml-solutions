def run_fedavg(client_partitions, test_features, test_labels, model_config, num_rounds, client_fraction, local_epochs, batch_size, learning_rate, seed):
    global_state = initialize_global_state(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
        seed
    )

    test_model = build_mlp_classifier(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
    )

    accuracies = []
    num_clients = len(client_partitions)
    clients_per_round = max(1, int(num_clients * client_fraction))

    for round_idx in range(num_rounds):
        rng = torch.Generator()
        rng.manual_seed(seed + round_idx)
        selected = torch.randperm(num_clients, generator=rng)[:clients_per_round].tolist()
        global_state = run_communication_round(
            global_state, client_partitions, selected, model_config,
            local_epochs, batch_size, learning_rate, seed + round_idx
        )

        load_model_state(test_model, global_state)
        test_model.eval()
        with torch.no_grad():
            logits = test_model(test_features)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == test_labels).float().mean().item()

        accuracies.append(accuracy)

    final_model = build_mlp_classifier(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
    )
    load_model_state(final_model, global_state)
    return final_model, accuracies
