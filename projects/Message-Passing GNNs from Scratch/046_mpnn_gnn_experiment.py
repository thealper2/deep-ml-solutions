def mpnn_gnn_experiment(num_nodes=40, num_features=8, num_classes=2, num_layers=3, hidden_dim=16, num_epochs=20, lr=0.01, seed=0):
    g = build_node_classification_dataset(1, num_nodes, num_classes, 0.5, 0.1, num_features, seed=seed)[0]
    x = g['node_features']
    edge_index = g['edge_index']
    y = g['node_labels']
    N = g['num_nodes']
    E = edge_index.shape[1]
    C = num_classes
    src, dst = edge_index[0], edge_index[1]

    torch.manual_seed(seed)
    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:N // 2]] = True
    dataset = {'x': x, 'edge_index': edge_index, 'y': y, 'train_mask': train_mask}

    dims = [(num_features if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]

    gcn_params = {}
    for i, (ind, outd) in enumerate(dims):
        p = init_gcn_parameters(ind, outd, with_bias=True, seed=seed + 10 + i)
        gcn_params[f'l{i}_weight'] = p['weight'].clone().detach().requires_grad_(True)
        gcn_params[f'l{i}_bias']   = p['bias'].clone().detach().requires_grad_(True)
    hp = init_gcn_parameters(hidden_dim, C, with_bias=True, seed=seed + 50)
    gcn_params['head_weight'] = hp['weight'].clone().detach().requires_grad_(True)
    gcn_params['head_bias']   = hp['bias'].clone().detach().requires_grad_(True)

    def gcn_forward(params, x, edge_index):
        s, d = edge_index[0], edge_index[1]
        param_list = [{'weight': params[f'l{i}_weight'], 'bias': params[f'l{i}_bias']}
                      for i in range(num_layers)]
        emb, _ = gcn_stack_forward(x, s, d, param_list,
                                   activations=[torch.relu] * num_layers, num_nodes=x.shape[0])
        return node_classification_head(emb, params['head_weight'], params['head_bias'])

    gat_params = {}
    for i, (ind, outd) in enumerate(dims):
        h0 = init_gat_parameters(ind, outd, num_heads=1, with_bias=True, seed=seed + 100 + i)[0]
        gat_params[f'l{i}_h0_weight']   = h0['weight'].clone().detach().requires_grad_(True)
        gat_params[f'l{i}_h0_attn_src'] = h0['attn_src'].clone().detach().requires_grad_(True)
        gat_params[f'l{i}_h0_attn_dst'] = h0['attn_dst'].clone().detach().requires_grad_(True)
        gat_params[f'l{i}_h0_bias']     = h0['bias'].clone().detach().requires_grad_(True)
    hp2 = init_gcn_parameters(hidden_dim, C, with_bias=True, seed=seed + 150)
    gat_params['head_weight'] = hp2['weight'].clone().detach().requires_grad_(True)
    gat_params['head_bias']   = hp2['bias'].clone().detach().requires_grad_(True)

    def gat_forward(params, x, edge_index):
        s, d = edge_index[0], edge_index[1]
        layer_param_list = [[{
            'weight':   params[f'l{i}_h0_weight'],
            'attn_src': params[f'l{i}_h0_attn_src'],
            'attn_dst': params[f'l{i}_h0_attn_dst'],
            'bias':     params[f'l{i}_h0_bias'],
        }] for i in range(num_layers)]
        emb, _ = gat_stack_forward(x, s, d, layer_param_list,
                                   merge_modes=['concat'] * num_layers,
                                   activations=[torch.relu] * num_layers, num_nodes=x.shape[0])
        return node_classification_head(emb, params['head_weight'], params['head_bias'])

    gcn_res = train_node_classifier(gcn_params, dataset, gcn_forward, num_epochs, lr)
    gat_res = train_node_classifier(gat_params, dataset, gat_forward, num_epochs, lr)

    with torch.no_grad():
        gcn_pl = [{'weight': gcn_res['params'][f'l{i}_weight'], 'bias': gcn_res['params'][f'l{i}_bias']}
                  for i in range(num_layers)]
        _, gcn_outs = gcn_stack_forward(x, src, dst, gcn_pl,
                                        activations=[torch.relu] * num_layers, num_nodes=N)
        gcn_os = oversmoothing_diagnostic(gcn_outs)

        gat_lpl = [[{'weight': gat_res['params'][f'l{i}_h0_weight'],
                     'attn_src': gat_res['params'][f'l{i}_h0_attn_src'],
                     'attn_dst': gat_res['params'][f'l{i}_h0_attn_dst'],
                     'bias': gat_res['params'][f'l{i}_h0_bias']}] for i in range(num_layers)]
        _, gat_outs = gat_stack_forward(x, src, dst, gat_lpl, merge_modes=['concat'] * num_layers,
                                        activations=[torch.relu] * num_layers, num_nodes=N)
        gat_os = oversmoothing_diagnostic(gat_outs)

    return {
        'gcn': {'history': gcn_res['history'], 'oversmoothing': gcn_os},
        'gat': {'history': gat_res['history'], 'oversmoothing': gat_os},
        'dataset_sizes': {'N': int(N), 'E': int(E), 'C': int(C)},
    }
