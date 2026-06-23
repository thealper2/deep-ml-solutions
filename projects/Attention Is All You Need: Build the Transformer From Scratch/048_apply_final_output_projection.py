def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)
