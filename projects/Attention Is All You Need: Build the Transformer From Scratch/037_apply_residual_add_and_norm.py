import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    combined = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined, gamma, beta, eps)
