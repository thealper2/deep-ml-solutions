def world_model_report(dataset: dict, params: dict, n_heads: int, n_rows: int, n_tokens: int,
                       max_pairs: int, n_trials: int, seed: int = 0) -> dict:
    G = dataset['G']
    T = dataset['tokens'].shape[1]

    vmr = valid_move_rate(dataset, params, n_heads, n_rows)

    H = eval_hidden_states(dataset, params, n_heads, n_rows)
    er = effective_rank(H)

    sc = sequence_compression(dataset, params, n_heads, n_tokens, max_pairs)

    dr = detour_robustness(params, n_heads, G, max_steps=T - 3, n_trials=n_trials, detour_prob=0.75, seed=seed)

    return {
        'valid_move_rate': round(float(vmr), 4),
        'effective_rank': round(float(er), 4),
        'sequence_compression': round(float(sc), 4),
        'detour_robustness': round(float(dr), 4),
    }
