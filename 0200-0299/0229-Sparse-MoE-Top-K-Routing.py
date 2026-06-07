import numpy as np

def moe_topk_routing(
    router_logits: np.ndarray,
    expert_outputs: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.
    
    For each token:
    1. Select the top-k experts based on router_logits
    2. Compute softmax weights over only the selected experts
    3. Return weighted combination of the selected expert outputs
    
    Args:
        router_logits: Shape (batch_size, num_experts)
                      Raw scores from the router for each expert
        expert_outputs: Shape (batch_size, num_experts, hidden_dim)
                       Output from each expert for each input
        k: Number of experts to select per token
        
    Returns:
        Shape (batch_size, hidden_dim) - weighted combination of expert outputs
    """
    num_tokens, num_experts = router_logits.shape

    top_k_indices = np.argsort(-router_logits, axis=1)[:, :k]

    top_k_logits = np.take_along_axis(router_logits, top_k_indices, axis=1)
    exp_logits = np.exp(top_k_logits - np.max(top_k_logits, axis=1, keepdims=True))
    weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True) 

    outputs = np.zeros((num_tokens, expert_outputs.shape[-1]))
    for i in range(num_tokens):
        for j in range(k):
            expert_idx = top_k_indices[i, j]
            outputs[i] += weights[i, j] * expert_outputs[i, expert_idx]

    return outputs.tolist()
