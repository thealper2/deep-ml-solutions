def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    derivation = (
        "For cross-entropy loss L = -1/B * sum_i log(p_i) where p_i = exp(logits_i) / sum(exp(logits_j)),\n"
        "the derivative of the log-softmax with respect to logits gives: ∂p_k/∂logits_j = p_k(δ_kj - p_j).\n"
        "Applying chain rule: ∂L/∂logits_j = -1/B * sum_i (1/p_i) * ∂p_i/∂logits_j = -1/B * sum_i (δ_ij - p_j) = (p_j - onehot_j) / B.\n"
        "Thus, dL/dlogits = (probs - onehot(targets)) / B."
    )
    return derivation
