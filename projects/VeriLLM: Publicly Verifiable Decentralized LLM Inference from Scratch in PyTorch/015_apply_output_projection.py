def apply_output_projection(context, attn_params):
    return linear_projection(context, attn_params['Wo'], attn_params['bo'])
