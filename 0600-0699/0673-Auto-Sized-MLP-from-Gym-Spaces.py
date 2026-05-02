def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.

    Args:
        obs_space: dict with 'type' and type-specific keys defining the observation space
        act_space: dict with 'type' and type-specific keys defining the action space
        hidden_layers: list of ints specifying hidden layer widths

    Returns:
        dict with keys:
          'input_dim': int
          'output_dim': int
          'layer_shapes': list of ((weight_rows, weight_cols), (bias_dim,)) tuples
          'total_params': int
    """
    if obs_space['type'] == 'Box':
        input_dim = 1
        for dim in obs_space['shape']:
            input_dim *= dim
    elif obs_space['type'] == 'Discrete':
        input_dim = obs_space['n']
    elif obs_space['type'] == 'MultiBinary':
        input_dim = obs_space['n']
    else:
        raise ValueError(f"Unknown observation space type: {obs_space['type']}")

    if act_space['type'] == 'Box':
        output_dim = 1
        for dim in act_space['shape']:
            output_dim *= dim
    elif act_space['type'] == 'Discrete':
        output_dim = act_space['n']
    elif act_space['type'] == 'MultiBinary':
        output_dim = act_space['n']
    else:
        raise ValueError(f"Unknown action space type: {act_space['type']}")

    dims = [input_dim] + hidden_layers + [output_dim]

    layer_shapes = []
    total_params = 0

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        weight_shape = (in_dim, out_dim)
        bias_shape = (out_dim,)
        layer_shapes.append((weight_shape, bias_shape))
        total_params += in_dim * out_dim + out_dim

    return {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'layer_shapes': layer_shapes,
        'total_params': total_params
    }