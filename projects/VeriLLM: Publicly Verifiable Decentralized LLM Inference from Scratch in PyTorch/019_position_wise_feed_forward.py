def position_wise_feed_forward(x, ffn_params):
    h = ffn_first_layer_gelu(x, ffn_params)
    return ffn_second_layer(h, ffn_params)
