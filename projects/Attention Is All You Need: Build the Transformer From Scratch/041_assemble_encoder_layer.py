def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    x = encoder_layer_self_attention_sublayer(
        x,
        layer_params['w_q'],
        layer_params['w_k'],
        layer_params['w_v'],
        layer_params['w_o'],
        layer_params['attn_gamma'],
        layer_params['attn_beta'],
        num_heads,
        src_mask,
    )
    x = encoder_layer_feed_forward_sublayer(
        x,
        layer_params['w1'],
        layer_params['b1'],
        layer_params['w2'],
        layer_params['b2'],
        layer_params['ffn_gamma'],
        layer_params['ffn_beta'],
    )
    return x
