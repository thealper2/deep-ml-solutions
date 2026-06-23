def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
        
    return x
