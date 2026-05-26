import numpy as np

def moe_load_balancing_loss(gate_logits, num_experts, alpha=0.01):
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: numpy array of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    max_logits = np.max(gate_logits, axis=1, keepdims=True)
    exp_logits = np.exp(gate_logits - max_logits)
    gate_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    num_tokens = gate_probs.shape[0]
    selected_experts = np.argmax(gate_probs, axis=1)
    f = np.zeros(num_experts)
    for i in range(num_experts):
        f[i] = np.mean(selected_experts == i)

    P = np.mean(gate_probs, axis=0)
    loss = num_experts * np.sum(f * P) * alpha
    return round(loss, 4)