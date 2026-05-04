import numpy as np

def rl_budget_loss(
    rewards: np.ndarray,
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    response_lengths: np.ndarray,
    token_budget: int,
    kl_coef: float,
    budget_penalty_coef: float
) -> float:
    """
    Compute the budget-constrained RL loss.
    
    The loss combines:
    1. Budget penalty for responses exceeding token_budget
    2. Advantage estimation (adjusted reward - baseline)
    3. KL regularization between current and old policy
    
    Loss formula: E[(advantage - kl_term)^2]
    
    Args:
        rewards: Shape (batch_size, K) - rewards for K samples per prompt
        log_probs: Shape (batch_size, K) - log π_θ(y|x) current policy
        old_log_probs: Shape (batch_size, K) - log π_old(y|x) old policy
        response_lengths: Shape (batch_size, K) - token lengths of responses
        token_budget: Maximum allowed tokens before penalty
        kl_coef: τ coefficient for KL regularization
        budget_penalty_coef: λ coefficient for budget penalty
        
    Returns:
        Scalar loss value (float)
    """
    penalty = -budget_penalty_coef * np.maximum(0, response_lengths - token_budget)
    adjusted_rewards = rewards + penalty
    baseline = np.mean(adjusted_rewards, axis=1, keepdims=True)
    advantages = adjusted_rewards - baseline
    kl_term = kl_coef * (log_probs - old_log_probs)
    loss = np.mean((advantages - kl_term) ** 2)
    return float(loss)