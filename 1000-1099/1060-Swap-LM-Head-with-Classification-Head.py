import numpy as np

def swap_lm_head_with_classifier(hidden_states, lm_head_weight, lm_head_bias, num_classes, seed=123):
    """
    Replace an LM head with a classification head and compute new logits.
    """
    batch_size, num_tokens, emb_dim = hidden_states.shape
    vocab_size = lm_head_weight.shape[0]

    np.random.seed(seed)
    new_weight = np.random.randn(num_classes, emb_dim) * 0.02
    new_bias = np.zeros(num_classes)

    logits = hidden_states @ new_weight.T + new_bias

    old_head_params = vocab_size * emb_dim + vocab_size
    new_head_params = num_classes * emb_dim + num_classes
    param_delta = new_head_params - old_head_params

    return {
        'output_shape': [batch_size, num_tokens, num_classes],
        'new_head_params': new_head_params,
        'old_head_params': old_head_params,
        'param_delta': param_delta,
        'logits': np.round(logits, 6).tolist(),
    }