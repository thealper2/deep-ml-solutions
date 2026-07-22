import numpy as np

def acceptance_rate_vs_temperature(draft_logits: np.ndarray, target_logits: np.ndarray, temperatures: np.ndarray) -> list:
    """
    Compute speculative decoding expected acceptance rate at various temperatures.
    
    Args:
        draft_logits: Logits from draft model, shape (vocab_size,)
        target_logits: Logits from target model, shape (vocab_size,)
        temperatures: Array of temperature values to evaluate
    
    Returns:
        List of acceptance rates (floats rounded to 4 decimal places)
    """
    draft_logits = np.array(draft_logits)
    target_logits = np.array(target_logits)
    temperatures = np.array(temperatures)
    
    acceptance_rates = []
    
    for T in temperatures:
        draft_exp = np.exp(draft_logits / T - np.max(draft_logits / T))
        target_exp = np.exp(target_logits / T - np.max(target_logits / T))
        draft_probs = draft_exp / np.sum(draft_exp)
        target_probs = target_exp / np.sum(target_exp)
        acceptance = np.sum(np.minimum(draft_probs, target_probs))
        acceptance_rates.append(round(float(acceptance), 4))
    
    return acceptance_rates
