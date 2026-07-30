import numpy as np

def grammar_constrained_step(logits, allowed, transitions, current_state, sampled_token):
    probs = [0.0] * len(logits)
    max_logit = max(logits[i] for i in allowed)
    exp_sum = 0.0
    exp_vals = {}

    for idx in allowed:
        exp_val = np.exp(logits[idx] - max_logit)
        exp_vals[idx] = float(exp_val)
        exp_sum += exp_val

    for idx in allowed:
        probs[idx] = float(exp_vals[idx] / exp_sum)
    
    next_state = transitions.get((current_state, sampled_token), current_state)
    return probs, next_state
