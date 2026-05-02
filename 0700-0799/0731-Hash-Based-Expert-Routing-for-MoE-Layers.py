import numpy as np

def hash_moe_forward(token_ids, embeddings, expert_weights, num_experts: int) -> np.ndarray:
    """
    Hash-based routing forward pass for an MoE layer.

    Args:
        token_ids: 1D array-like of N integer token IDs
        embeddings: 2D array-like of shape (N, d) of token embeddings
        expert_weights: list of E numpy arrays, each of shape (d, d_out)
        num_experts: number of experts E

    Returns:
        numpy array of shape (N, d_out) with per-token expert outputs.
    """
    token_ids = np.array(token_ids)
    embeddings = np.array(embeddings)

    N = len(token_ids)
    d_out = expert_weights[0].shape[1]

    expert_indices = token_ids % num_experts
    output = np.zeros((N, d_out))

    for i, expert_idx in enumerate(expert_indices):
        output[i] = embeddings[i] @ expert_weights[expert_idx]

    return output