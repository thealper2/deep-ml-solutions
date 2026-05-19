import numpy as np

def alphazero_loss(p_logits: np.ndarray, v_pred: np.ndarray, pi: np.ndarray, z: np.ndarray) -> float:
    """
    Compute the AlphaZero policy + value loss.

    Args:
        p_logits: (batch, num_actions) policy logits
        v_pred:   (batch,) predicted values
        pi:       (batch, num_actions) MCTS target distributions (rows sum to 1)
        z:        (batch,) target outcomes

    Returns:
        Scalar loss value.
    """
    value_loss = np.mean((v_pred - z) ** 2)
    max_logits = np.max(p_logits, axis=1, keepdims=True)
    log_softmax = p_logits - max_logits - np.log(np.sum(np.exp(p_logits - max_logits), axis=1, keepdims=True))
    policy_loss = -np.mean(np.sum(pi * log_softmax, axis=1))
    return value_loss + policy_loss