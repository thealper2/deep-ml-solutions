def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    obs = dataset['observations']
    embeddings = encode_batch(obs, encoder_params).detach()
    states = dataset['states'].float()

    if probe_params is None:
        probe_params = init_linear_probe(embeddings.shape[1], states.shape[1], seed=0)

    inner = {'w': probe_params['w'].T.contiguous(), 'b': probe_params['b']}
    t = train_linear_probe(embeddings, states, inner, num_probe_steps)
    trained = {'w': t['w'].T.contiguous(), 'b': t['b']}

    with torch.no_grad():
        pred_states = embeddings @ trained['w'].T + trained['b']
        mse = float(torch.mean((pred_states - states) ** 2).item())
        mean_abs_error = float(torch.mean(torch.abs(pred_states - states)).item())

    return {'mse': mse, 'mean_abs_error': mean_abs_error, 'probe_params': trained}

