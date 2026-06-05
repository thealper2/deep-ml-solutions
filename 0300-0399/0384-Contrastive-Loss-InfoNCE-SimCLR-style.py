import numpy as np

def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Array of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    sim_matrix = embeddings @ embeddings.T / temperature
    n = len(embeddings)
    loss = 0.0

    for i in range(n):
        if i % 2 == 0:
            pos_idx = i + 1
        else:
            pos_idx = i - 1

        pos_sim = sim_matrix[i, pos_idx]
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        denom = np.sum(np.exp(sim_matrix[i, mask]))
        loss_i = -pos_sim + np.log(denom)
        loss += loss_i

    return round(loss / n, 4)