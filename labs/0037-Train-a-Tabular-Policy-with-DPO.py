import numpy as np

def train(train_chosen, train_rejected, n_items, beta=0.5):
    """
    Train a preference policy with DPO on unpaired item indices.

    A frozen uniform reference over `n_items` catalog items is assumed.
    You only observe pairwise preferences (chosen_idx, rejected_idx).

    Args:
        train_chosen: np.ndarray[int] shape (N,) — preferred item indices
        train_rejected: np.ndarray[int] shape (N,) — rejected item indices
        n_items: size of the discrete item catalog
        beta: DPO temperature

    Returns:
        score: callable(indices: np.ndarray[int]) -> np.ndarray[float]
               Higher score = more preferred. Used to evaluate whether
               score(chosen) > score(rejected) on held-out pairs.
    """
    logits = np.zeros(n_items, dtype=np.float64)
    lr = 0.1
    epochs = 500
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_chosen))
        chosen = train_chosen[indices]
        rejected = train_rejected[indices]
        logsumexp = np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits)
        log_probs = logits - logsumexp
        log_p_chosen = log_probs[chosen]
        log_p_rejected = log_probs[rejected]
        diff = log_p_chosen - log_p_rejected
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(beta * diff, -50, 50)))
        grad = np.zeros(n_items, dtype=np.float64)
        for i in range(len(chosen)):
            c = chosen[i]
            r = rejected[i]
            grad[c] -= beta * (1 - sigmoid[i])
            grad[r] += beta * (1 - sigmoid[i])
        
        logits -= lr * grad
        logits -= np.mean(logits)
    
    logsumexp = np.log(np.sum(np.exp(logits - np.max(logits)))) + np.max(logits)
    log_probs = logits - logsumexp
    
    def score(indices):
        indices = np.asarray(indices, dtype=int)
        return log_probs[indices]
    
    return score
