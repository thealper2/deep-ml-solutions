def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    norm_out = layernorm_forward_affine(x, ln_params['gamma'], ln_params['beta'], eps=1e-5)
    sublayer_out = sublayer_fn(norm_out['y'], sublayer_params)
    y = residual_forward(x, sublayer_out['y'])
    cache = {
        'x': x,
        'ln_cache': norm_out['cache'],
        'sublayer_cache': sublayer_out['cache'],
    }
    return {'y': y, 'cache': cache}
