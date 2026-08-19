import math
import torch

def train_tiny_mamba_and_generate(corpus, n_steps, lr, prompt, max_new_tokens, d_model=16, n_layers=2, d_state=4, d_inner=32, conv_kernel=3, seed=0):
    """Train a tiny character-level Mamba LM on corpus and greedily generate from prompt."""
    torch.manual_seed(seed)
    
    vocab = sorted(set(corpus))
    char_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_char = {i: c for i, c in enumerate(vocab)}
    V = len(vocab)
    
    corpus_ids = torch.tensor([char_to_id[c] for c in corpus], dtype=torch.long).unsqueeze(0)
    prompt_ids = torch.tensor([char_to_id[c] for c in prompt], dtype=torch.long)
    
    params = {}
    
    embed_weight = torch.randn(V, d_model) * 0.02
    embed_weight.requires_grad_(True)
    params['embed_weight'] = embed_weight
    
    lm_head_weight = torch.randn(V, d_model) * 0.02
    lm_head_weight.requires_grad_(True)
    params['lm_head_weight'] = lm_head_weight
    
    norm_weight = torch.ones(d_model)
    norm_weight.requires_grad_(True)
    params['norm_weight'] = norm_weight
    
    blocks = []
    for _ in range(n_layers):
        block = {}
        
        block['norm_weight'] = torch.ones(d_model)
        block['norm_weight'].requires_grad_(True)
        
        block['in_proj_weight'] = torch.randn(2 * d_inner, d_model) * 0.02
        block['in_proj_weight'].requires_grad_(True)
        block['in_proj_bias'] = torch.zeros(2 * d_inner)
        block['in_proj_bias'].requires_grad_(True)
        
        block['conv_weight'] = torch.randn(d_inner, conv_kernel) * 0.02
        block['conv_weight'].requires_grad_(True)
        block['conv_bias'] = torch.zeros(d_inner)
        block['conv_bias'].requires_grad_(True)
        
        block['dt_weight'] = torch.randn(d_inner, d_inner) * 0.02
        block['dt_weight'].requires_grad_(True)
        block['dt_bias'] = torch.zeros(d_inner)
        block['dt_bias'].requires_grad_(True)
        
        block['weight_b'] = torch.randn(d_state, d_inner) * 0.02
        block['weight_b'].requires_grad_(True)
        block['weight_c'] = torch.randn(d_state, d_inner) * 0.02
        block['weight_c'].requires_grad_(True)
        
        log_a = torch.zeros(d_inner, d_state)
        for i in range(d_inner):
            for n in range(d_state):
                log_a[i, n] = math.log(n + 1)
        log_a.requires_grad_(True)
        block['log_a'] = log_a
        
        block['out_proj_weight'] = torch.randn(d_model, d_inner) * 0.02
        block['out_proj_weight'].requires_grad_(True)
        block['out_proj_bias'] = torch.zeros(d_model)
        block['out_proj_bias'].requires_grad_(True)
        
        blocks.append(block)
    
    params['blocks'] = blocks
    
    losses = []
    for _ in range(n_steps):
        loss = sgd_training_step(corpus_ids, params, lr)
        losses.append(loss)
    
    generated_ids = greedy_generate(prompt_ids, params, max_new_tokens)
    generated_text = ''.join(id_to_char[int(id_.item())] for id_ in generated_ids)
    
    return generated_text, losses