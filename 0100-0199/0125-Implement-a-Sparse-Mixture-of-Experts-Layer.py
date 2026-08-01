import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    batch_size, seq_len, d_model = x.shape
    num_experts = We.shape[0]
    logits = x @ Wg
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    gate_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    top_k_indices = np.argsort(-gate_probs, axis=-1)[:, :, :top_k]
    top_k_probs = np.take_along_axis(gate_probs, top_k_indices, axis=-1)
    top_k_probs_normalized = top_k_probs / np.sum(top_k_probs, axis=-1, keepdims=True)
    output = np.zeros_like(x)
    for b in range(batch_size):
        for s in range(seq_len):
            token = x[b, s]
            for k in range(top_k):
                expert_idx = top_k_indices[b, s, k]
                weight = top_k_probs_normalized[b, s, k]
                expert_out = token @ We[expert_idx]
                output[b, s] += weight * expert_out

    return output
