def initialize_global_state(input_size, hidden_size, num_classes, seed):
    torch.manual_seed(seed)
    model = build_mlp_classifier(input_size, hidden_size, num_classes)
    return clone_model_state(model)
