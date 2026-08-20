import jax
import jax.numpy as jnp

def sample_pair(seed, shape):
    """Return (a, b): two different standard-normal arrays of `shape`,
    drawn from two subkeys split off PRNGKey(seed)."""
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2 = jax.random.split(key, 2)
    sample1 = jax.random.normal(subkey1, shape)
    sample2 = jax.random.normal(subkey2, shape)
    return sample1, sample2