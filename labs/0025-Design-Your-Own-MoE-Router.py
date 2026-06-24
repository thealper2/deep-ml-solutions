import numpy as np

def route(x, W_gate):
    '''
    Route inputs to experts in a Mixture of Experts layer.
    
    Args:
        x: numpy array of shape (batch_size, hidden_dim)
           Hidden activations from the previous layer
        W_gate: numpy array of shape (hidden_dim, num_experts)
           Learnable gate weight matrix (32 experts)
    
    Returns:
        numpy array of shape (batch_size, num_experts)
        Per-expert weights that are:
        - Non-negative
        - Sum to ~1.0 per sample (row)
        - Based on the input x and gate weights W_gate
    
    The harness combines expert outputs as:
        output = sum(weights[:, i] * expert_i(x) for each expert i)
    
    Each expert is small (128->16), so routing quality matters!
    With 32 experts, naive strategies that spread weight everywhere
    perform poorly. The key insight from modern LLMs is SPARSE
    routing: only activate a few experts per input.
    
    Hint: Compute softmax scores, keep only the top-k experts,
    zero out the rest, and renormalize.
    '''
    batch_size = x.shape[0]
    num_experts = W_gate.shape[1]
    
    logits = x @ W_gate

    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(logits, axis=1, keepdims=True)

    k = 2
    top_k_indices = np.argsort(-probs, axis=1)[:, :k]

    weights = np.zeros_like(probs)
    for i in range(batch_size):
        weights[i, top_k_indices[i]] = probs[i, top_k_indices[i]]

    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-8, row_sums)
    weights = weights / row_sums
    
    return weights
