import torch

def init_gpt_params(vocab_size: int, d_model: int, n_layers: int, max_len: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    std = 0.02
    
    params = {}
    
    params['wte'] = torch.randn(vocab_size, d_model) * std
    params['wte'].requires_grad_(True)
    
    params['wpe'] = torch.randn(max_len, d_model) * std
    params['wpe'].requires_grad_(True)
    
    for l in range(n_layers):
        params[f'ln1_w{l}'] = torch.ones(d_model)
        params[f'ln1_w{l}'].requires_grad_(True)
        params[f'ln1_b{l}'] = torch.zeros(d_model)
        params[f'ln1_b{l}'].requires_grad_(True)
        
        params[f'qkv_w{l}'] = torch.randn(d_model, 3 * d_model) * std
        params[f'qkv_w{l}'].requires_grad_(True)
        params[f'qkv_b{l}'] = torch.zeros(3 * d_model)
        params[f'qkv_b{l}'].requires_grad_(True)
        
        params[f'proj_w{l}'] = torch.randn(d_model, d_model) * std
        params[f'proj_w{l}'].requires_grad_(True)
        params[f'proj_b{l}'] = torch.zeros(d_model)
        params[f'proj_b{l}'].requires_grad_(True)
        
        params[f'ln2_w{l}'] = torch.ones(d_model)
        params[f'ln2_w{l}'].requires_grad_(True)
        params[f'ln2_b{l}'] = torch.zeros(d_model)
        params[f'ln2_b{l}'].requires_grad_(True)
        
        params[f'fc_w{l}'] = torch.randn(d_model, 4 * d_model) * std
        params[f'fc_w{l}'].requires_grad_(True)
        params[f'fc_b{l}'] = torch.zeros(4 * d_model)
        params[f'fc_b{l}'].requires_grad_(True)
        
        params[f'fc2_w{l}'] = torch.randn(4 * d_model, d_model) * std
        params[f'fc2_w{l}'].requires_grad_(True)
        params[f'fc2_b{l}'] = torch.zeros(d_model)
        params[f'fc2_b{l}'].requires_grad_(True)
    
    params['lnf_w'] = torch.ones(d_model)
    params['lnf_w'].requires_grad_(True)
    params['lnf_b'] = torch.zeros(d_model)
    params['lnf_b'].requires_grad_(True)
    
    params['head_w'] = torch.randn(d_model, vocab_size) * std
    params['head_w'].requires_grad_(True)
    params['head_b'] = torch.zeros(vocab_size)
    params['head_b'].requires_grad_(True)
    
    return params