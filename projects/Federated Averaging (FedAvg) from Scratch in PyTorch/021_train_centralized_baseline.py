def train_centralized_baseline(train_features, train_labels, test_features, test_labels, model_config, num_epochs, batch_size, learning_rate, seed):
    model = build_mlp_classifier(
        model_config['input_size'],
        model_config['hidden_size'],
        model_config['num_classes'],
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        batches = iterate_client_batches(train_features, train_labels, batch_size, seed + epoch)
        for batch_features, batch_labels in batches:
            local_sgd_step(model, optimizer, batch_features, batch_labels)

    return evaluate_accuracy(model, test_features, test_labels)
