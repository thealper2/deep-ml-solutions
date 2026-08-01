import numpy as np

def moe_shared_expert_forward(
    X: np.ndarray,
    W_gate: np.ndarray,
    W_shared: np.ndarray,
    W_experts: list,
    top_k: int = 2
) -> dict:
    """
    Forward pass of a Mixture of Experts layer with a shared expert.
    
    Args:
        X: Input tokens, shape (num_tokens, d_model)
        W_gate: Gating weights, shape (d_model, num_routed_experts)
        W_shared: Shared expert weights, shape (d_model, d_out)
        W_experts: List of routed expert weight matrices, each (d_model, d_out)
        top_k: Number of routed experts per token
    
    Returns:
        Dictionary with keys: 'output', 'shared_output', 'routed_output',
                              'routing_indices', 'routing_weights'
    """
    num_tokens, d_model = X.shape
    num_routed_experts = len(W_experts)
    d_out = W_experts[0].shape[1]
    
    gate_logits = X @ W_gate
    max_logits = np.max(gate_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(gate_logits - max_logits)
    gate_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    top_k_indices = np.zeros((num_tokens, top_k), dtype=int)
    top_k_probs = np.zeros((num_tokens, top_k))
    
    for t in range(num_tokens):
        probs = gate_probs[t]
        sorted_pairs = sorted([(probs[i], i) for i in range(num_routed_experts)], 
                            key=lambda x: (-x[0], -x[1]))
        for k in range(top_k):
            top_k_indices[t, k] = sorted_pairs[k][1]
            top_k_probs[t, k] = sorted_pairs[k][0]
    
    top_k_probs_normalized = top_k_probs / np.sum(top_k_probs, axis=-1, keepdims=True)
    
    shared_output = X @ W_shared
    routed_output = np.zeros((num_tokens, d_out))
    
    for t in range(num_tokens):
        token = X[t]
        for k in range(top_k):
            expert_idx = top_k_indices[t, k]
            weight = top_k_probs_normalized[t, k]
            expert_out = token @ W_experts[expert_idx]
            routed_output[t] += weight * expert_out
    
    output = shared_output + routed_output
    
    return {
        'output': output,
        'shared_output': shared_output,
        'routed_output': routed_output,
        'routing_indices': top_k_indices,
        'routing_weights': top_k_probs_normalized
    }
