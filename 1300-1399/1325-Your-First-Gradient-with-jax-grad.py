import jax
import jax.numpy as jnp

def df_dx(x):
    """Derivative of f(x) = x**3 + 2x at float x, via jax.grad. Returns float."""
    f = lambda x: x**3 + 2 * x
    grad_f = jax.grad(f)
    return float(grad_f(x))
