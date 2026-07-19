def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    conv1_w = encoder_params['conv1_w']
    conv1_b = encoder_params['conv1_b']
    conv2_w = encoder_params['conv2_w']
    conv2_b = encoder_params['conv2_b']
    fc_w = encoder_params['fc_w']
    fc_b = encoder_params['fc_b']

    x = torch.nn.functional.conv2d(obs, conv1_w, conv1_b, stride=1, padding=1)
    x = torch.relu(x)

    x = torch.nn.functional.conv2d(x, conv2_w, conv2_b, stride=2, padding=1)
    x = torch.relu(x)

    x = x.view(x.size(0), -1)
    x = x @ fc_w.T + fc_b
    return x
