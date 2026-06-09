import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    weights = jax.random.normal(key, shape=(in_dim, out_dim)) * scale
    biases = jnp.zeros((out_dim,))
    return {'W': weights, 'b': biases}
