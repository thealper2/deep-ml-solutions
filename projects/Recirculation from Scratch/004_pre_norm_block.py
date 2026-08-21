def pre_norm_block(x, block):
    """Wrap attention and feed-forward as a pre-norm residual transformer block."""
    x_norm = rms_norm(x, block['attn_gain'])
    attn_out = causal_self_attention(x_norm, block['w_q'], block['w_k'], block['w_v'], block['w_o'])
    x = x + attn_out
    x_norm = rms_norm(x, block['ffn_gain'])
    ffn_out = gelu_ffn(x_norm, block['w_ff1'], block['w_ff2'])
    x = x + ffn_out
    return x