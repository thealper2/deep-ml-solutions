import jax.numpy as jnp

def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    return x @ layer_params['W'] + layer_params['b']
