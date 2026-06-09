import jax
import jax.numpy as jnp


def make_prng_key(seed):
    return jax.random.PRNGKey(seed)
