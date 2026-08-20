import jax
import jax.numpy as jnp

def dot(a, b):
    """Dot product of two 1-D vectors (given)."""
    return jnp.dot(a, b)

def batched_dot(A, B):
    """Row-wise dot products of (N, D) matrices A and B via jax.vmap. Returns shape (N,)."""
    batched_dot_fn = jax.vmap(dot)
    return batched_dot_fn(A, B)