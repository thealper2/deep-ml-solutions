import jax
import jax.numpy as jnp

def per_sample_grads(w, b, X, y):
    """Per-example gradients of (w·x + b - y)^2.
    Returns (dW, db) with shapes (N, D) and (N,)."""
    def loss(w, b, x, y_i):
        pred = jnp.dot(w, x) + b
        return (pred - y_i) ** 2

    grad_loss = jax.grad(loss, argnums=(0, 1))
    vmap_grad = jax.vmap(grad_loss, in_axes=(None, None, 0, 0))
    dW, db = vmap_grad(w, b, X, y)
    return dW, db