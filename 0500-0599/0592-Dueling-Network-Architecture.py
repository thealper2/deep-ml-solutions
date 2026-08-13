import numpy as np

def dueling_network_forward(x, shared_weights, shared_bias,
                           value_weights, value_bias,
                           advantage_weights, advantage_bias,
                           aggregation='mean'):
    """
    Forward pass of a Dueling Network Architecture.
    
    Args:
        x: Input features, shape (batch_size, input_dim)
        shared_weights: Shared layer weights, shape (input_dim, hidden_dim)
        shared_bias: Shared layer bias, shape (hidden_dim,)
        value_weights: Value stream weights, shape (hidden_dim, 1)
        value_bias: Value stream bias, shape (1,)
        advantage_weights: Advantage stream weights, shape (hidden_dim, num_actions)
        advantage_bias: Advantage stream bias, shape (num_actions,)
        aggregation: 'mean' or 'max' for advantage centering
    
    Returns:
        Q-values as numpy array, shape (batch_size, num_actions)
    """
    shared_features = np.maximum(0, np.dot(x, shared_weights) + shared_bias)
    state_values = np.dot(shared_features, value_weights) + value_bias
    advantages = np.dot(shared_features, advantage_weights) + advantage_bias
    if aggregation == 'mean':
        agg_advantages = np.mean(advantages, axis=-1, keepdims=True)
    elif aggregation == 'max':
        agg_advantages = np.max(advantages, axis=-1, keepdims=True)

    q_values = state_values + (advantages - agg_advantages)
    return q_values
