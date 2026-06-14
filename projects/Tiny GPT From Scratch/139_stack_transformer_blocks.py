import numpy as np

def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts.

    Each block dict has keys 'ln1', 'attn', 'ln2', 'ffn'.
    """
    blocks = []
    d_head = d_model // n_heads

    for layer_idx in range(n_layers):
        ln1 = {
            'gamma': np.ones(d_model),
            'beta': np.zeros(d_model),
        }
        ln2 = {
            'gamma': np.ones(d_model),
            'beta': np.zeros(d_model),
        }

        attn = {
            'Wq': scale_w_small(make_2d_random(d_model, d_model, seed=0), 0.02),
            'Wk': scale_w_small(make_2d_random(d_model, d_model, seed=1), 0.02),
            'Wv': scale_w_small(make_2d_random(d_model, d_model, seed=2), 0.02),
            'Wo': scale_w_small(make_2d_random(d_model, d_model, seed=3), 0.02),
            'bo': np.zeros(d_model),
        }

        ffn = {
            'W1': scale_w_small(make_2d_random(d_model, d_ff, seed=4), 0.02),
            'b1': np.zeros(d_ff),
            'W2': scale_w_small(make_2d_random(d_ff, d_model, seed=5), 0.02),
            'b2': np.zeros(d_model),
        }

        block = {
            'ln1': ln1,
            'attn': attn,
            'ln2': ln2,
            'ffn': ffn,
        }

        blocks.append(block)

    return blocks
