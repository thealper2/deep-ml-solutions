import numpy as np

def residual_add_and_norm(x, sublayer_output, ln_params, eps=1e-5):
    combined = x + sublayer_output
    return layer_norm_apply(combined, ln_params, eps)
