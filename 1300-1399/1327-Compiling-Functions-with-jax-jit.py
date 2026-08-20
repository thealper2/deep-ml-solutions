import jax
import jax.numpy as jnp

@jax.jit
def standardize(x):
    """Return (x - mean(x)) / std(x) for a 1-D array x."""
    mean = jnp.mean(x)
    std = jnp.std(x)
    return (x - mean) / std
