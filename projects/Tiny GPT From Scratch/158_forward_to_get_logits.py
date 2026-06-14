def forward_to_get_logits(params, context_ids):
    """Run the full model forward and return only the logits tensor."""
    B, T = context_ids.shape
    d_model = params['tok_emb'].shape[1]

    tok_emb = params['tok_emb'][context_ids]
    pos_emb = params['pos_emb'][:T, :]

    x = tok_emb + pos_emb
    
    for block_params in params['blocks']:
        out = transformer_block_forward(x, block_params)
        x = out['y']

    y, _ = final_layernorm_forward(x, params['ln_f']['gamma'], params['ln_f']['beta'])
    logits = y @ params['lm_head']['w_lm'] + params['lm_head']['b_lm']
    return logits
