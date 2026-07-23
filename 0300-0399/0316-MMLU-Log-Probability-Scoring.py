import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    log_probs = np.array(log_probs)
    correct_answers = np.array(correct_answers)
    predictions = np.argmax(log_probs, axis=1)
    accuracy = np.mean(predictions == correct_answers)
    shifted_log_probs = log_probs - np.max(log_probs, axis=-1, keepdims=True)
    exp_probs = np.exp(shifted_log_probs)
    probabilities = exp_probs / np.sum(exp_probs, axis=-1, keepdims=True)
    row_indices = np.arange(len(correct_answers))
    correct_probabilities = probabilities[row_indices, correct_answers]
    avg_correct_prob = np.mean(correct_probabilities)
    return {
        'accuracy': round(float(accuracy), 4),
        'predictions': predictions.tolist(),
        'avg_correct_prob': round(float(avg_correct_prob), 4),
    }
