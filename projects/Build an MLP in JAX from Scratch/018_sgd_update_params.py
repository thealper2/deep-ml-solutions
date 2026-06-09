import jax
import jax.numpy as jnp

def sgd_update_params(params, grads, learning_rate):
    new_params = []
    for layer_params, layer_grads in zip(params, grads):
        new_W = layer_params['W'] - learning_rate * layer_grads['W']
        new_b = layer_params['b'] - learning_rate * layer_grads['b']
        new_params.append({'W': new_W, 'b': new_b})
    
    return new_params
