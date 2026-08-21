def adaptive_recirculate(s, d, mixer):
    """Token-conditional vector mix of matched source into destination."""
    concat_sd = concat_residuals(s, d)
    alpha, beta = vector_mix_mlp(concat_sd, mixer)
    mixed = hadamard_mix(s, d, alpha, beta)
    return mixed
