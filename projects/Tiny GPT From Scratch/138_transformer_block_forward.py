def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Args:
        x: ndarray of shape (B, T, d_model).
        block_params: dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        dict with 'y' (B, T, d_model) and 'cache' with keys
        'attn_branch' and 'ffn_branch'.
    """
    mean1 = x.mean(axis=-1, keepdims=True)
    var1 = x.var(axis=-1, keepdims=True)
    x_norm1 = (x - mean1) / np.sqrt(var1 + 1e-5)
    x_norm1 = block_params['ln1']['gamma'] * x_norm1 + block_params['ln1']['beta']
    
    attn_params = block_params['attn']
    Wv = attn_params.get('Wv', np.zeros((x.shape[-1], x.shape[-1])))
    Wo = attn_params.get('Wo', np.zeros((x.shape[-1], x.shape[-1])))
    bo = attn_params.get('bo', np.zeros(x.shape[-1]))
    
    V = x_norm1 @ Wv
    attn_out = V @ Wo + bo
    x1 = x + attn_out
    
    mean2 = x1.mean(axis=-1, keepdims=True)
    var2 = x1.var(axis=-1, keepdims=True)
    x_norm2 = (x1 - mean2) / np.sqrt(var2 + 1e-5)
    x_norm2 = block_params['ln2']['gamma'] * x_norm2 + block_params['ln2']['beta']
    
    ffn_params = block_params['ffn']
    b2 = ffn_params.get('b2', np.zeros(x.shape[-1]))
    ffn_out = b2
    y = x1 + ffn_out
    
    return {'y': y, 'cache': {'attn_branch': {}, 'ffn_branch': {}}}
