def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    y = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params['w_q_self'],
        layer_params['w_k_self'],
        layer_params['w_v_self'],
        layer_params['w_o_self'],
        layer_params['self_gamma'],
        layer_params['self_beta'],
        num_heads,
        tgt_mask
    )
    
    y = decoder_layer_cross_attention_sublayer(
        y,
        encoder_output,
        layer_params['w_q_cross'],
        layer_params['w_k_cross'],
        layer_params['w_v_cross'],
        layer_params['w_o_cross'],
        layer_params['cross_gamma'],
        layer_params['cross_beta'],
        num_heads,
        src_mask
    )
    
    y = decoder_layer_feed_forward_sublayer(
        y,
        layer_params['w1'],
        layer_params['b1'],
        layer_params['w2'],
        layer_params['b2'],
        layer_params['ffn_gamma'],
        layer_params['ffn_beta']
    )
    
    return y
