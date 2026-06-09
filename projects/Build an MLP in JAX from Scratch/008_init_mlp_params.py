import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    weights = jax.random.normal(key, shape=(in_dim, out_dim)) * scale
    biases = jnp.zeros((out_dim,))
    return {'W': weights, 'b': biases}

def init_mlp_params(key, layer_sizes, scale=0.1):
    params = []
    keys = jax.random.split(key, num=len(layer_sizes) - 1)

    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        layer_params = init_linear_layer(keys[i], in_dim, out_dim, scale)
        params.append(layer_params)

    return params
