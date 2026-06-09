import jax
import jax.numpy as jnp

def compute_param_grads(params, x, one_hot_targets):
    grad_fn = jax.grad(loss_fn_of_params, argnums=0)
    return grad_fn(params, x, one_hot_targets)
