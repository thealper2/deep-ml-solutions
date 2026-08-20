import jax
import jax.numpy as jnp

def sgd_step(params, grads, lr):
    """Return a new PyTree: params - lr * grads, leaf-wise (params unchanged)."""
    def update(p, g):
        return p - lr * g

    return jax.tree_util.tree_map(update, params, grads)